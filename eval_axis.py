import os
import numpy as np
from utils.general_utils import safe_state, build_rotation
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scipy.spatial.transform import Rotation
from utils.system_utils import mkdir_p
from utils.rotation_utils import *
import glob
import cv2
import json
from itertools import permutations

def line_distance(a_o, a_d, b_o, b_d):
    normal = np.cross(a_d, b_d)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:   # parallel
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    else:
        return np.abs(np.dot(normal, a_o - b_o)) / normal_length

def eval_axis_and_state(axis_a, axis_b, joint_type='revolute', reverse=False):
    a_d, b_d = axis_a['axis_direction'], axis_b['axis_direction']

    angle = np.rad2deg(np.arccos(np.dot(a_d, b_d) / np.linalg.norm(a_d) / np.linalg.norm(b_d)))
    angle = min(angle, 180 - angle)

    if joint_type == 'revolute':
        a_o, b_o = axis_a['axis_position'], axis_b['axis_position']
        distance = line_distance(a_o, a_d, b_o, b_d)

        a_r, b_r = axis_a['rotation'], axis_b['rotation']
        if reverse:
            a_r = a_r.T

        r_diff = np.matmul(a_r, b_r.T)
        state = np.rad2deg(np.arccos(np.clip((np.trace(r_diff) - 1.0) * 0.5, a_min=-1, a_max=1)))
    else:
        distance = 0
        a_t, b_t = axis_a['translation'], axis_b['translation']
        if reverse:
            a_t = -a_t

        state = np.linalg.norm(a_t - b_t)

    return angle, distance, state

def interpret_transforms(base_R, base_t, R, t, joint_type='revolute'):
    """
    base_R, base_t, R, t are all from canonical to world
    rewrite the transformation = global transformation (base_R, base_t) {R' part + t'} --> s.t. R' and t' happens in canonical space
    R', t':
    - revolute: R'p + t' = R'(p - a) + a, R' --> axis-theta representation; axis goes through a = (I - R')^{-1}t'
    - prismatic: R' = I, t' = l * axis_direction
    """
    R = np.matmul(base_R.T, R)
    t = np.matmul(base_R.T, (t - base_t).reshape(3, 1)).reshape(-1)

    if joint_type == 'revolute':
        rotvec = Rotation.from_matrix(R).as_rotvec()
        theta = np.linalg.norm(rotvec, axis=-1)
        axis_direction = rotvec / max(theta, (theta < 1e-8))
        try:
            axis_position = np.matmul(np.linalg.inv(np.eye(3) - R), t.reshape(3, 1)).reshape(-1)
        except:   # TO DO find the best solution
            axis_position = np.zeros(3)
        axis_position += axis_direction * np.dot(axis_direction, -axis_position)
        joint_info = {'axis_position': axis_position,
                      'axis_direction': axis_direction,
                      'theta': np.rad2deg(theta),
                      'rotation': R, 'translation': t}

    elif joint_type == 'prismatic':
        theta = np.linalg.norm(t)
        axis_direction = t / max(theta, (theta < 1e-8))
        joint_info = {'axis_direction': axis_direction, 'axis_position': np.zeros(3), 'theta': theta,
                      'rotation': R, 'translation': t}

    return joint_info, R, t

def read_gt(gt_path):
    with open(gt_path, 'r') as f:
        info = json.load(f)

    all_trans_info = info['trans_info']
    if isinstance(all_trans_info, dict):
        all_trans_info = [all_trans_info]
    ret_list = []
    for trans_info in all_trans_info:
        axis = trans_info['axis']
        axis_o, axis_d = np.array(axis['o']), np.array(axis['d'])
        R_coord = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]).T
        axis_o = np.matmul(R_coord, axis_o)
        axis_d = np.matmul(R_coord, axis_d)
        axis_type = trans_info['type']
        l, r = trans_info[axis_type]['l'], trans_info[axis_type]['r']

        if axis_type == 'rotate':
            rotvec = axis_d * np.deg2rad(r - l)
            rot = R.from_rotvec(rotvec).as_matrix()
            trans = np.matmul(np.eye(3) - rot, axis_o.reshape(3, 1)).reshape(-1)
            joint_type = 'revolute'
        else:
            rot = np.eye(3)
            trans = (r - l) * axis_d
            joint_type = 'prismatic'
        ret_list.append({'axis_position': axis_o, 'axis_direction': axis_d, 'theta': r - l, 'joint_type': axis_type, 'rotation': rot, 'translation': trans,
                         'type': joint_type})
    return ret_list

def axis_metrics(motion, gt):
    # pred axis
    pred_axis_d = motion['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt['axis_d']
    gt_axis_o = gt['axis_o']
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err

def eval_axis(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    source_path = dataset.source_path
    
    gt_joints = read_gt(os.path.join(source_path, 'gt', 'trans.json'))

    model_path = dataset.model_path
    find_latest_iter = 0
    a = 10000
    b = 10000
    c = 10000
    it = 0
    best_parts_results = []
    
    for ck_iter in glob.glob(os.path.join(model_path, 'ckpts/*.pth')):
        iter = int(os.path.basename(ck_iter).split('.')[0].split('_')[-1])


        ckpt = torch.load(os.path.join(model_path, 'ckpts', f'ours_{iter}.pth'))
        articulation_params = ckpt['articulation_params']
        art_R = articulation_params["art_R"].cuda()
        art_T = articulation_params["art_T"].cuda()
        articulation_matrix = build_rotation(art_R)
        N = art_R.shape[0]
        
        angle_mean = 0
        distance_mean = 0
        theta_diff_mean = 0
        current_parts_results = []

        
        for i in range(0, N-1):
            cur_R = articulation_matrix[i].detach().cpu().numpy()
            cur_T = art_T[i].detach().cpu().numpy()
            if np.abs(cur_R - np.eye(3)).mean() < 5e-2:
                joint_type = 'prismatic'
            else:
                joint_type = 'revolute'
            pred_joint, cur_rot, cur_trans = interpret_transforms(np.eye(3), np.zeros(3), 
                                                                cur_R, cur_T, joint_type=joint_type)
            angle, distance, theta_diff = eval_axis_and_state(pred_joint, gt_joints[i-1], joint_type=joint_type,)
            angle_mean += angle
            distance_mean += distance * 10
            theta_diff_mean += theta_diff
            
            current_parts_results.append((angle, distance*10, theta_diff))
                
        angle_mean /= (N-1)
        distance_mean /= (N-1)
        theta_diff_mean /= (N-1)
        
        if angle_mean < a and distance_mean < b or angle_mean < a and theta_diff_mean < c or distance_mean < b and theta_diff_mean < c:
            a = angle_mean
            b = distance_mean
            c = theta_diff_mean
            it = iter
            best_parts_results = current_parts_results.copy()
            
        del current_parts_results
    print("The best:", it)
    print("Angle mean:", a)
    print("Distance mean:", b)
    print("Theta diff mean:", c)
    
    file = os.path.join(model_path, 'results.txt')
    mkdir_p(os.path.dirname(file))
    with open(file, 'w') as f:
        f.write(f"The best: {it}\n")
        f.write(f"Parts num: {N}\n")
        f.write(f"Angle mean: {a}\n")
        f.write(f"Distance mean: {b}\n")
        f.write(f"Theta diff mean: {c}\n")
        
        for idx, (angle, distance, theta_diff) in enumerate(best_parts_results):
            f.write(f"Part {idx}: angle={angle:.4f}, distance={distance:.4f}, theta_diff={theta_diff:.4f}\n")
        #f.write(f"test")
        
        
    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    eval_axis(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)