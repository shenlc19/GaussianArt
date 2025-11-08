
import cv2
import copy
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state, build_rotation, vis_depth
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import numpy as np
from moviepy.editor import VideoFileClip
import json
from matplotlib.image import imsave
import time
import open3d as o3d
from utils.rotation_utils import rotation_matrices_to_axes_angles, R_from_axis_angle

color_map = {
    0: [0, 0, 0],
    1: [232, 126, 23],
    2: [58, 228, 24],
    3: [55, 18, 238],
    4: [255, 255, 0],
    5: [165, 42, 42],
    6: [238, 130, 238],
    7: [255, 248, 220],
}

def generate_video(imgs, video_name, fps=15, brighten=False):
    # imgs: list of img tensors [3, H, W]
    height, width = imgs[0].shape[1], imgs[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for img in imgs:
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if brighten:
            img[:, :width//2] = cv2.convertScaleAbs(img[:, :width//2], alpha=0.8, beta=1)
        video.write(img)
    video.release()


def generate_camera_poses(N=30):
    """
    Generate camera poses around the scene center on a circle.

    Parameters:
    - r: Radius of the circle.
    - theta: Elevation angle in degrees.
    - num_samples: Number of samples (camera positions) to generate.

    Returns:
    - poses: A list of camera poses (4x4 transformation matrices).
    """
    poses = []
    traj_info = {
        "radius": 3.5,
        "theta": [-0, 0.1],
        "d_theta": 0.2,
        "phi": [0, 2],
        "d_phi": -0.75,
        "rotx90": 0,
        "roty180": 0
    }
    radius, r_theta, r_phi = traj_info['radius'], traj_info['theta'], traj_info['phi']
    d_theta, d_phi = traj_info['d_theta'], traj_info['d_phi']

    thetas = np.linspace(r_theta[0] * np.pi, r_theta[1] * np.pi, N)
    thetas = np.concatenate([np.zeros(N//2), thetas]) + d_theta * np.pi
    azimuths = np.linspace(r_phi[0] * np.pi, r_phi[1] * np.pi, N)
    azimuths = np.concatenate([np.zeros(N//2), azimuths]) + d_phi * np.pi
    roty180 = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]]) if traj_info['roty180'] else np.eye(4)
    rotx90 = np.array([[1, 0, 0, 0],
                       [0, 0, -1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]]) if traj_info['rotx90'] else np.eye(4)
    
    for theta, azimuth in zip(thetas, azimuths):
        # Convert spherical coordinates to Cartesian coordinates

        x = radius * np.cos(azimuth) * np.cos(theta)
        y = radius * np.sin(azimuth) * np.cos(theta)
        z = radius * np.sin(theta)

        # Camera position
        position = np.array([x, y, z])

        # Compute the forward direction (pointing towards the origin)
        forward = position / np.linalg.norm(position)

        # Compute the right and up vectors for the camera coordinate system
        up = np.array([0, 0, 1])
        if np.allclose(forward, up) or np.allclose(forward, -up):
            up = np.array([0, 1, 0])
        right = np.cross(up, forward)
        up = np.cross(forward, right)

        # Normalize the vectors
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        # Construct the rotation matrix
        rotation_matrix = np.vstack([right, up, forward]).T

        # Construct the transformation matrix (4x4)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        transformation_matrix = roty180 @ rotx90.T @ transformation_matrix

        poses.append(transformation_matrix)
    return poses


def generate(views, N=30):
    new_views = []
    poses = generate_camera_poses(N)
    for i, pose in enumerate(poses):
        view = copy.deepcopy(views[0])
        view.fid = i / (len(poses) - 1)
        view.gt_alpha_mask = None
        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]
        view.reset_extrinsic(R, T)
        new_views.append(view)
    return new_views

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, extract_mesh=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")
    rgbd_path = os.path.join(model_path, name, "ours_{}".format(iteration), "rgbd")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    makedirs(semantic_path, exist_ok=True)
    makedirs(rgbd_path, exist_ok=True)

    ckpt_path = os.path.join(model_path, "ckpts/ours_{}.pth".format(iteration))
    ckpt = torch.load(ckpt_path)

    articulation_params = ckpt["articulation_params"]
    art_R = articulation_params["art_R"].cuda()
    art_T = articulation_params["art_T"].cuda()
    N = art_R.shape[0]

    # encoder_state_dict = ckpt["encoder_state_dict"]
    # embed_fn, input_ch = get_embedder(4, 0)
    # encoder = WeightEncoder(D=8, W=256,
    #                         input_ch=input_ch, output_ch=N, skips=[4])
    # encoder.load_state_dict(encoder_state_dict)
    # encoder = encoder.cuda()

    # articulation_weights = run_network(gaussians.get_xyz.detach(), embed_fn, encoder)
    articulation_weights = gaussians.get_weight
    max_indices = torch.argmax(articulation_weights, dim=1)
    new_articulation_weights = torch.zeros_like(articulation_weights)
    new_articulation_weights[torch.arange(articulation_weights.shape[0]), max_indices] = 1
    articulation_weights = new_articulation_weights
    articulation_matrix = build_rotation(art_R)
    articulation_trans = art_T

    weights_save_dir = os.path.join(model_path, 'articulation_weights')
    makedirs(weights_save_dir, exist_ok=True)
    torch.save(articulation_weights, os.path.join(weights_save_dir, f'{iteration}.pth'))

    points = gaussians.get_xyz.detach().cpu().numpy()
    N = articulation_matrix.shape[0]
    color_labels = torch.rand((N, 3)).float().cuda()
    if articulation_matrix.shape[0] <= 3:
        color_labels = torch.tensor([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])[:N, ...].cuda()
    color = torch.einsum('ni,ij->nj', articulation_weights, color_labels).detach().cpu().numpy()
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points)
    colored_pcd.colors = o3d.utility.Vector3dVector(color)

    scene_name = os.path.basename(model_path) if 'art' not in model_path else os.path.basename(os.path.dirname(model_path))

    # o3d.io.write_point_cloud("output/weights_vis_{}.ply".format(scene_name), colored_pcd)

    axis, angle = rotation_matrices_to_axes_angles(articulation_matrix)

    rgbmaps = []
    depthmaps = []
    
    t_list = []
    
    N_rot_views = len(views) // 3
    ts = torch.cat([torch.linspace(0, 1, N_rot_views // 2), torch.linspace(1, 0, N_rot_views // 2)]*3)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        t = ts[idx]
        angle_t = angle * t
        articulation_trans_t = articulation_trans * t
        articulation_matrix_t = torch.stack([R_from_axis_angle(axis[i], angle_t[i]) for i in range(angle_t.shape[0])]).cuda()
        torch.cuda.synchronize(); t0 = time.time()
        render_pkg = render(view, gaussians, pipeline, background, 
                           articulation_weights=articulation_weights, articulation_matrix=articulation_matrix_t, articulation_trans=articulation_trans_t, force_transform=True)
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        depth = render_pkg["depth"]
        semantic = render_pkg["weight"]
        rgbmaps.append(rendering)
        depthmaps.append(depth)
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{view.time:.0f}_{idx:04d}.png"))
        
    rgbds = []
    for i in range(len(rgbmaps)):
        rgb = rgbmaps[i].cpu() # [3, H, W]
        torchvision.utils.save_image(rgb, os.path.join(render_path, '{0:05d}'.format(i) + ".png")) 
        depth = vis_depth(depthmaps[i], os.path.join(depth_path, '{0:05d}'.format(i) + ".png")) # [3, H, W]
        rgbd = torchvision.utils.make_grid([rgb, depth], nrow=2, padding=0)
        torchvision.utils.save_image(rgbd, os.path.join(rgbd_path, '{0:05d}'.format(i) + ".png"))
        rgbds.append(rgbd.cpu())
    
    generate_video(rgbds, os.path.join(model_path, name, "ours_{}".format(iteration), f"{scene_name}.mp4"))
    print(f"Saved video to {os.path.join(model_path, name, 'ours_{}'.format(iteration), f'{scene_name}.mp4')}")
               
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, N_frames: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        cam_traj = scene.getTrainCameras()
        cam_traj = generate(cam_traj, N=N_frames)
        
        render_set(dataset.model_path, dataset.source_path, "video", scene.loaded_iter, cam_traj, gaussians, pipeline, background)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--image_name", type=str, default="0001")
    parser.add_argument("--N_frames", default=60, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(os.path.join(args.model_path, 'results.txt'), 'r') as f:
        iter_line = f.readline()
        iteration = int(iter_line.split(':')[-1])
        
    args.iteration = iteration

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.N_frames)
        