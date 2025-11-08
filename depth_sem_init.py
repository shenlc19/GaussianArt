#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, build_rotation, convert_cam_name
from utils.articulation_utils import *
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.system_utils import mkdir_p
from utils.rotation_utils import *
import open3d as o3d
import random
import glob
from utils.loss_utils import reproj_modified

# color_map = {
#     -1: [c / 255 for c in [255, 255, 255]], # white 
#     0: [c / 255 for c in [232, 126, 23]], # orange    
#     1: [c / 255 for c in [58, 228, 24]], # green  
#     2: [c / 255 for c in [55, 18, 238]], # blue     
#     3: [c / 255 for c in [255, 255, 0]], # yellow
#     4: [c / 255 for c in [165, 42, 42]], # brown
#     5: [c / 255 for c in [238, 130, 238]], # pink
#     6: [c / 255 for c in [255, 248, 220]], # beige
# }
color_map = {
    -1: [c / 255 for c in [255, 255, 255]], # white 
    0: [c / 255 for c in [255, 0, 0]], # orange    
    1: [c / 255 for c in [0, 255, 0]], # green  
    2: [c / 255 for c in [55, 18, 238]], # blue     
    3: [c / 255 for c in [255, 255, 0]], # yellow
    4: [c / 255 for c in [165, 42, 42]], # brown
    5: [c / 255 for c in [238, 130, 238]], # pink
    6: [c / 255 for c in [255, 248, 220]], # beige
}

def reproj(depth_map, intrinsics, w2c, depth_mask=None):
    def pix2ndc(v, S):
        return (v * 2.0 + 1.0) / S - 1.0
    
    projectinverse = intrinsics.T.inverse()
    camera2world = w2c.T.inverse()
    
    depth_map = torch.tensor(depth_map, device='cuda')
    width, height = depth_map.squeeze().shape[1], depth_map.squeeze().shape[0]

    x_grid, y_grid = torch.meshgrid(torch.arange(height).cuda().float(), 
                                    torch.arange(width).cuda().float(),
                                    )
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    
    ndcu, ndcv = pix2ndc(x_grid, depth_map.squeeze().shape[0]), pix2ndc(y_grid, depth_map.squeeze().shape[1])
    
    if depth_mask is not None:
        ndcu = ndcu[depth_mask.reshape(-1)]
        ndcv = ndcv[depth_mask.reshape(-1)]

    ndcu = ndcu.unsqueeze(-1)
    ndcv = ndcv.unsqueeze(-1)
    ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1)
    localpointuv = ndccamera @ projectinverse.T
    diretioninlocal = localpointuv / localpointuv[:,3:]

    if depth_mask is not None:
        depth_map = depth_map[depth_mask]
    
    targetPz = depth_map.reshape(-1).unsqueeze(-1)
    rate = targetPz / diretioninlocal[:, 2:3]
    localpoint = diretioninlocal * rate
    localpoint[:, -1] = 1
    worldpointH = localpoint.float() @ camera2world.T.float()
    worldpoint = worldpointH / worldpointH[:, 3:]

    return worldpoint

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(3)
        scene = Scene(dataset, gaussians, load_iteration=None, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        init_points = []
        init_rgbs = []
        init_semantics = []
        
        cam_dict = {convert_cam_name(cam.image_name, cam.time): cam for cam in scene.getTrainCameras()}

        all_start_ids = []
        for cam in cam_dict:
            if cam.startswith("0"):
                all_start_ids.append(int(cam.split('_')[-1]))
        sel_start_ids = random.sample(all_start_ids, 100)
        # sel_start_ids = all_start_ids

        for i in tqdm(sel_start_ids):
            view_id = f"{i:05d}"
            view = cam_dict[view_id]
            depth_map = view.depth.cuda()
            rgb_map = view.original_image.cuda()
            if torch.all(rgb_map == 0):
                continue
            semantic_map = view.semantic.cuda()
            semantic_rgb = torch.zeros((semantic_map.shape[0], semantic_map.shape[1], 3), device='cuda')
            for label, color in color_map.items():
                color_tensor = torch.tensor(color, device='cuda')
                semantic_rgb[semantic_map == label] = color_tensor
            semantic_mask = (semantic_map >= 0)
            depth_mask = (depth_map > 0)
            semantic_depth_mask = torch.logical_and(semantic_mask, depth_mask)
            # x_grid, y_grid = torch.meshgrid(torch.arange(800).cuda().float(), 
            #                             torch.arange(800).cuda().float(),
            #                             )
            # x_grid = x_grid.reshape(-1)
            # y_grid = y_grid.reshape(-1)
            # pixs = torch.stack([y_grid, x_grid], dim=-1)
            points = reproj(depth_map=depth_map, intrinsics=view.projection_matrix, w2c=view.world_view_transform,
                            depth_mask=semantic_depth_mask)[:, :3]
            if points.shape[0] == 0:
                continue
            rgbs = rgb_map[..., semantic_depth_mask].reshape(3, -1).transpose(1, 0)
            semantic_rgb = semantic_rgb[semantic_depth_mask].reshape(-1, 3)
            semantic_map = semantic_map[semantic_depth_mask].reshape(-1)
            # sample_mask = torch.randint(0, len(points) - 1, (1000, ))
            sample_mask = torch.randint(0, len(points) - 1, (500, ))
            init_points.append(points[sample_mask, ...])
            # init_rgbs.append(semantic_rgb[sample_mask, ...])
            init_rgbs.append(rgbs[sample_mask, ...])
            init_semantics.append(semantic_map[sample_mask])
        
        init_points = torch.cat(init_points, dim=0)
        init_rgbs = torch.cat(init_rgbs, dim=0)
        init_semantics = torch.cat(init_semantics, dim=0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(init_points.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(init_rgbs.detach().cpu().numpy())
        pcd_semantics = init_semantics.detach().cpu().numpy()
        
        # downsampled_pcd = pcd.farthest_point_down_sample(num_samples=100000)
        o3d.io.write_point_cloud(os.path.join(args.source_path, 
                                              "points3d.ply"), pcd)
        np.save(os.path.join(args.source_path, "semantics.npy"), pcd_semantics)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.data_device = "cuda"

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)