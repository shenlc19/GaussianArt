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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from scene.gaussian_model import GaussianModel
from torchvision.utils import save_image
from utils.articulation_utils import run_network
import numpy as np
from pytorch3d.ops import ball_query, knn_points, knn_gather
from pytorch3d.structures.pointclouds import Pointclouds
from typing import Union
from tqdm import tqdm
from utils.general_utils import build_rotation
import os

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def knn(point_cloud, k):
    # Compute pairwise distances
    distances = torch.cdist(point_cloud, point_cloud, p=2)
    
    # Get the k smallest distances for each point, ignoring the point itself (distance 0)
    try:
        knn_distances, knn_indices = torch.topk(distances, k=k, largest=False)
    except:
        breakpoint()
    
    # Exclude the point itself by slicing the first column
    knn_indices = knn_indices[:, 1:]
    knn_distances = knn_distances[:, 1:]
    
    return knn_indices, knn_distances

def knn_indices_single_point(point_cloud, point, k):
    # 计算目标点与其他所有点之间的欧氏距离
    diffs = point_cloud - point.unsqueeze(0)  # 形状为 (N, 3)
    dists = torch.norm(diffs, dim=1)  # 形状为 (N,)
    
    # 使用topk找到距离最小的k个点的索引
    _, indices = torch.topk(dists, k=k, largest=False)
    
    return indices

def knn_loss(gaussians : GaussianModel, articulation_weight, m=None):
    pc = gaussians.get_xyz
    N = pc.shape[0]
    if m is None:
        m = N // 3000
    random_indices = torch.randint(0, N, (m, )).cuda()
    accumulated_knn_loss = 0.
    k = 50
    for ind in random_indices:
        point = pc[ind, ...]
        sel_indices = knn_indices_single_point(pc, point, k=k)
        accumulated_knn_loss += torch.abs(articulation_weight[sel_indices, ...] - \
                                articulation_weight[ind, ...].unsqueeze(0)).mean()
    accumulated_knn_loss /= m
    
    return accumulated_knn_loss

def arap_loss(gaussians : GaussianModel, articulation_weight, articulation_matrix, articulation_t, m=None):
    pc = gaussians.get_xyz
    N = pc.shape[0]
    if m is None:
        m = N // 3000
    random_indices = torch.randint(0, N, (m, )).cuda()
    accumulated_arap_loss = 0.
    k = 50
    for ind in random_indices:
        point = pc[ind, ...]
        sel_indices = knn_indices_single_point(pc, point, k=k)
        sel_neighbor_points = pc[sel_indices, ...]
        sel_weights = articulation_weight[sel_indices, ...]
        sel_trans = torch.einsum('nj, jx -> nx', sel_weights, articulation_t)
        sel_rot = torch.einsum('nj, jrc -> nrc', sel_weights, articulation_matrix)
        local_arap_loss = (point[None, ...] - sel_neighbor_points + sel_trans[:1, ...] - sel_trans - 
                                  torch.einsum('nij, nj -> ni', sel_rot, point[None, ...] - sel_neighbor_points))
        accumulated_arap_loss += local_arap_loss.abs().mean()
        
    accumulated_arap_loss /= m
    
    return accumulated_arap_loss

def reproj(depth_map, intrinsics, w2c, depth_mask=None, pixs=None):
    def pix2ndc(v, S):
        return (v * 2.0 + 1.0) / S - 1.0
    
    projectinverse = intrinsics.T.inverse()
    camera2world = w2c.T.inverse()
    
    depth_map = torch.tensor(depth_map, device='cuda')
    width, height = depth_map.squeeze().shape[1], depth_map.squeeze().shape[0]

    if depth_mask is None:
        x_grid, y_grid = torch.meshgrid(torch.arange(height).cuda().float(), 
                                        torch.arange(width).cuda().float(),
                                        )
        x_grid = x_grid.reshape(-1)
        y_grid = y_grid.reshape(-1)
    else:
        x_grid = depth_mask.nonzero()[..., 0]
        y_grid = depth_mask.nonzero()[..., 1]
    
    
    ndcu, ndcv = pix2ndc(x_grid, depth_map.squeeze().shape[0]), pix2ndc(y_grid, depth_map.squeeze().shape[1])
    
    # if depth_mask is not None:
    #     ndcu = ndcu[depth_mask.reshape(-1)]
    #     ndcv = ndcv[depth_mask.reshape(-1)]

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
    worldpointH = localpoint @ camera2world.T
    worldpoint = worldpointH / worldpointH[:, 3:]

    return worldpoint

def reproj_modified(depth_map, intrinsics, w2c, depth_mask=None, pixs=None):
    def pix2ndc(v, S):
        return (v * 2.0 + 1.0) / S - 1.0
    
    projectinverse = intrinsics.T.inverse()
    camera2world = w2c.T.inverse()
    
    depth_map = torch.tensor(depth_map, device='cuda')

    x_grid = pixs[..., 1].long()
    y_grid = pixs[..., 0].long()

    ndcu, ndcv = pix2ndc(x_grid, depth_map.squeeze().shape[0]), pix2ndc(y_grid, depth_map.squeeze().shape[1])
    
    # if depth_mask is not None:
    #     ndcu = ndcu[depth_mask.reshape(-1)]
    #     ndcv = ndcv[depth_mask.reshape(-1)]

    ndcu = ndcu.unsqueeze(-1)
    ndcv = ndcv.unsqueeze(-1)
    ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1)
    localpointuv = ndccamera @ projectinverse.T
    diretioninlocal = localpointuv / localpointuv[:,3:]

    depths = depth_map[x_grid, y_grid]
    
    targetPz = depths.reshape(-1).unsqueeze(-1)
    rate = targetPz / diretioninlocal[:, 2:3]
    localpoint = diretioninlocal * rate
    localpoint[:, -1] = 1
    worldpointH = localpoint @ camera2world.T
    worldpoint = worldpointH / worldpointH[:, 3:]

    return worldpoint

def pix2world(src_view, pixs):
    def pix2ndc(v, S):
        return (v * 2.0 + 1.0) / S - 1.0
    
    depth_map = src_view.depth
    intrinsics = src_view.projection_matrix
    w2c = src_view.world_view_transform
    
    projectinverse = intrinsics.T.inverse()
    camera2world = w2c.T.inverse()
    
    depth_map = torch.tensor(depth_map, device='cuda')
    height, width = depth_map.squeeze().shape[1], depth_map.squeeze().shape[0]

    xs, ys = pixs[..., 0].long(), pixs[..., 1].long()
    ndcu, ndcv = pix2ndc(xs, width), pix2ndc(ys, height)
    
    ndcu = ndcu.unsqueeze(-1)
    ndcv = ndcv.unsqueeze(-1)
    ndccamera = torch.cat((ndcv, ndcu,   torch.ones_like(ndcu) * (1.0) , torch.ones_like(ndcu)), 1)
    localpointuv = ndccamera @ projectinverse.T
    diretioninlocal = localpointuv / localpointuv[:,3:]
    
    targetPz = depth_map[ys, xs].reshape(-1).unsqueeze(-1)
    rate = targetPz / diretioninlocal[:, 2:3]
    localpoint = diretioninlocal * rate
    localpoint[:, -1] = 1
    worldpointH = localpoint @ camera2world.T
    worldpoint = worldpointH / worldpointH[:, 3:]

    return worldpoint

def pix2world_modified(src_view, pixs):
    depth_mask = None
    # depth_mask = torch.zeros_like(src_view.depth).cuda()
    # xs, ys = pixs[..., 0], pixs[..., 1]
    # depth_mask[ys.long(), xs.long()] = 1
    # depth_mask = depth_mask.bool()
    P_world = reproj_modified(src_view.depth, src_view.projection_matrix, src_view.world_view_transform, depth_mask=depth_mask, pixs=pixs)
    return P_world

def world2pix(P_world, tgt_view, h, w):
    def ndc2pix(v, S):
        return ((v + 1.0) * (S - 1) + 1) // 2

    p_hom = P_world @ tgt_view.full_proj_transform
    p_proj = p_hom / p_hom[..., -1:]
    
    x, y = ndc2pix(p_proj[..., 0], w).float(), ndc2pix(p_proj[..., 1], h).float()
    pred_tgt_pixels = torch.stack([x, y], dim=-1)
    return pred_tgt_pixels

# def forw_flow(src_view, src_pixels, tgt_view, tgt_pixels, embed_fn, encoder, articulation_matrix, articulation_trans):
#     P_world = pix2world_modified(src_view, src_pixels)[:, :3]
#     articulation_weights = run_network(P_world, embed_fn, encoder)

#     W = articulation_weights
#     T = articulation_trans
#     trans = torch.einsum('nj, jx -> nx', W, T)
#     R = articulation_matrix # R: K, 3, 3
#     frame = torch.einsum('nj, jrc -> nrc', W, R) # frame: N, 3, 3

#     transformed_P_world = torch.einsum('njk, nk -> nj', frame, P_world) + trans

#     # h, w = tgt_view.depth.squeeze().shape

#     # transformed_P_world_hom = torch.cat([transformed_P_world, torch.ones_like(transformed_P_world[..., :1])], dim=-1)

#     P_world_tgt = pix2world_modified(tgt_view, tgt_pixels)[:, :3]

#     return (P_world_tgt - transformed_P_world).abs().mean()

def forw_flow(src_view, src_pixels, tgt_view, tgt_pixels, articulation_weights, articulation_matrix, articulation_trans):
    P_world = pix2world_modified(src_view, src_pixels)[:, :3]
    # articulation_weights = run_network(P_world, embed_fn, encoder)

    W = articulation_weights
    T = articulation_trans
    trans = torch.einsum('nj, jx -> nx', W, T)
    R = articulation_matrix # R: K, 3, 3
    frame = torch.einsum('nj, jrc -> nrc', W, R) # frame: N, 3, 3

    transformed_P_world = torch.einsum('njk, nk -> nj', frame, P_world) + trans

    P_world_tgt = pix2world_modified(tgt_view, tgt_pixels)[:, :3]

    return (P_world_tgt - transformed_P_world).abs().mean()

def get_weights_gs(query_xyz, gs_xyz, gs_weights):
    knn_results = knn_points(query_xyz.unsqueeze(0), gs_xyz.unsqueeze(0), K=10)
    dists, idx = knn_results.dists.squeeze(), knn_results.idx.squeeze()
    query_weights = dists / torch.norm(dists, dim=-1, keepdim=True)
    weights = (gs_weights[idx] * query_weights).mean(dim=-1)

    return weights

def forw_flow_knn(src_view, src_pixels, tgt_view, tgt_pixels, articulation_matrix, articulation_trans, gs_xyz, gs_weights):
    P_world = pix2world_modified(src_view, src_pixels)[:, :3]
    articulation_weights = get_weights_gs(P_world, gs_xyz, gs_weights)

    pass

def calculate_valid_mask(pc0, pc1, k=10, threshold=0.02):
    knn_indices, _ = knn(pc0, k=k)
    corr_avg = pc1[knn_indices].mean(dim=1)
    large_deviation_mask = torch.norm(pc1 - corr_avg, dim=-1) < threshold
    return large_deviation_mask

def filter_corr(cam_dict, view_name):
    view = cam_dict[view_name]
    view_corr = view.correspondence
    for cur_corr in view_corr:
        # filter one batch of correspondence
        keys_list = sorted(list(cur_corr.keys()))
        # get viewpoint cams
        v0 = cam_dict[keys_list[0]]
        v1 = cam_dict[keys_list[1]]
        if v0.depth.nelement() == 0 or v1.depth.nelement() == 0:
            continue
        pix0 = torch.tensor(cur_corr[keys_list[0]]).float().cuda()
        pix1 = torch.tensor(cur_corr[keys_list[1]]).float().cuda()
        P_world_0 = pix2world_modified(v0, pix0)[:, :3]
        P_world_1 = pix2world_modified(v1, pix1)[:, :3]
        k = min(P_world_0.shape[0], 10)
        valid_mask = calculate_valid_mask(P_world_0, P_world_1, k=k)
        valid_mask = valid_mask.cpu().numpy()
        for key in keys_list:
            cur_corr[key] = cur_corr[key][valid_mask, ...]

def filter_all_corr(cam_dict):
    for frame in tqdm(cam_dict):
        filter_corr(cam_dict, frame)

# def corr_loss(cam_dict, view_corr, view_name, embed_fn, encoder, articulation_matrix, articulation_trans):
#     all_src_pixels = []
#     all_pred_tgt_pixels = []
#     all_tgt_pixels = []
#     l = 0.
#     for cur_corr in np.random.permutation(view_corr):
#         keys_list = sorted(list(cur_corr.keys()))
#         if len(cur_corr[keys_list[0]]) == 0:
#             continue
#         if keys_list[0].split('_')[0] == f'{0:05d}' and keys_list[1].split('_')[0] == f'{1:05d}':
#             src_view = cam_dict[keys_list[0]]
#             tgt_view = cam_dict[keys_list[1]]
#             if (src_view.depth.nonzero().sum() == 0) or (tgt_view.depth.nonzero().sum() == 0):
#                 return 0.
#             src_pixels = torch.tensor(cur_corr[keys_list[0]]).float().cuda()
#             tgt_pixels = torch.tensor(cur_corr[keys_list[1]]).float().cuda()
#             # forward src pixels to tgt pixels
#             l = forw_flow(src_view, src_pixels, tgt_view, tgt_pixels, embed_fn, encoder, articulation_matrix, articulation_trans)
#             break

#     # sz = torch.tensor(src_view.depth.squeeze().shape).cuda()

#     # return ((tgt_pixels - pred_tgt_pixels) / sz[None, :]).abs().mean()

#     return l

def corr_loss(cam_dict, view_corr, view_name, semantic, articulation_matrix, articulation_trans):
    all_src_pixels = []
    all_pred_tgt_pixels = []
    all_tgt_pixels = []
    l = 0.
    for cur_corr in np.random.permutation(view_corr):
        keys_list = sorted(list(cur_corr.keys()))
        if len(cur_corr[keys_list[0]]) == 0:
            continue
        if keys_list[0].split('_')[0] == f'{0:05d}' and keys_list[1].split('_')[0] == f'{1:05d}':
            src_view = cam_dict[keys_list[0]]
            tgt_view = cam_dict[keys_list[1]]
            if (src_view.depth.nonzero().sum() == 0) or (tgt_view.depth.nonzero().sum() == 0):
                return 0.
            src_pixels = torch.tensor(cur_corr[keys_list[0]]).float().cuda()
            tgt_pixels = torch.tensor(cur_corr[keys_list[1]]).float().cuda()
            semantics = []
            for i in range(src_pixels.shape[0]):
                pix_sem = semantic[:,:,src_pixels[i, 1].long(), src_pixels[i, 0].long()]
                semantics.append(pix_sem)        
            semantics = torch.cat(semantics, dim=0).float()
            
            # forward src pixels to tgt pixels
            l = forw_flow(src_view, src_pixels, tgt_view, tgt_pixels, semantics, articulation_matrix, articulation_trans)
            break
    return l

def get_corr_3d(cam_dict, view_name):
    def convert_name(orig_name):
        if 'start' in orig_name:
            state = 0
        else:
            state = 1
        frame_id = int(os.path.basename(orig_name).split('.')[0])
        return f'{state:05d}_{frame_id:03d}'
    view_corr = cam_dict[view_name].correspondence
    all_src_points = []
    all_tgt_points = []

    for cur_corr in np.random.permutation(view_corr):
        keys_list = sorted([convert_name(i) for i in list(cur_corr.keys())])
        if len(cur_corr[list(cur_corr.keys())[0]]) == 0:
            continue
        if keys_list[0].split('_')[0] == f'{0:05d}' and keys_list[1].split('_')[0] == f'{1:05d}':
            src_view = cam_dict[keys_list[0]]
            tgt_view = cam_dict[keys_list[1]]
            if (src_view.depth.nonzero().sum() == 0) or (tgt_view.depth.nonzero().sum() == 0):
                continue
            keys_list = sorted(list(cur_corr.keys()))
            src_pixels = torch.tensor(cur_corr[keys_list[0]]).float().cuda()
            tgt_pixels = torch.tensor(cur_corr[keys_list[1]]).float().cuda()
            src_P_world = pix2world_modified(src_view, src_pixels)[:, :3]
            tgt_P_world = pix2world_modified(tgt_view, tgt_pixels)[:, :3]

            all_src_points.append(src_P_world)
            all_tgt_points.append(tgt_P_world)
    
    all_src_points = torch.cat(all_src_points, dim=0)
    all_tgt_points = torch.cat(all_tgt_points, dim=0)
    
    return all_src_points, all_tgt_points

def log_loss(articulation_weights):
    eps = 1e-10
    return (-articulation_weights * torch.log(articulation_weights + eps) - \
            (1 - articulation_weights) * torch.log(1 - articulation_weights + eps)).mean()


def ellipsoid_covariance_intersection(centers, covariances, p1, p2):
    """
    Determine whether a batch of line segments intersects with a batch of ellipsoids
    represented by their covariance matrices.

    Args:
    - centers: Tensor of shape (N, 3) representing the centers of the ellipsoids.
    - covariances: Tensor of shape (N, 3, 3) representing the covariance matrices of the ellipsoids.
    - p1: Tensor of shape (M, 3) representing the starting points of the line segments.
    - p2: Tensor of shape (M, 3) representing the end points of the line segments.

    Returns:
    - intersection_mask: Boolean tensor of shape (M, N) where True indicates that the line segment intersects the ellipsoid.
    """
    
    # Number of ellipsoids (N) and line segments (M)
    N = centers.shape[0]
    M = p1.shape[0]
    
    # Vector from p1 to p2
    d = p2.unsqueeze(1) - p1.unsqueeze(1)  # Shape: (M, 1, 3)
    
    # Shift the line segment points to the ellipsoid's local coordinates
    p1_shifted = p1.unsqueeze(1) - centers.unsqueeze(0)  # Shape: (M, N, 3)
    
    # Inverse of the covariance matrices
    inv_covariances = torch.inverse(covariances)  # Shape: (N, 3, 3)
    
    # Apply inverse covariance matrix to the direction vector d and shifted points p1
    d_inv_cov = torch.einsum('nij,mnj->mni', inv_covariances, d)  # Shape: (M, N, 3)
    p1_shifted_inv_cov = torch.einsum('nij,mnj->mni', inv_covariances, p1_shifted)  # Shape: (M, N, 3)
    
    # Compute the coefficients A, B, C for the quadratic equation At^2 + Bt + C = 0
    A = torch.sum(d * d_inv_cov, dim=-1)  # Shape: (M, N)
    B = 2 * torch.sum(p1_shifted * d_inv_cov, dim=-1)  # Shape: (M, N)
    C = torch.sum(p1_shifted * p1_shifted_inv_cov, dim=-1) - 1  # Shape: (M, N)
    
    # Compute the discriminant: Delta = B^2 - 4AC
    discriminant = B ** 2 - 4 * A * C  # Shape: (M, N)
    
    # Check if there are any real solutions (intersection occurs if discriminant >= 0)
    has_real_solutions = discriminant >= 0
    
    # Solve for t using the quadratic formula (only where discriminant >= 0)
    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0))  # Clamp to avoid NaNs
    t1 = (-B - sqrt_discriminant) / (2 * A)  # Shape: (M, N)
    t2 = (-B + sqrt_discriminant) / (2 * A)  # Shape: (M, N)
    
    # Check if t1 or t2 falls within the range [0, 1] (i.e., line segment intersects)
    intersects = ((t1 >= 0) & (t1 <= 1)) | ((t2 >= 0) & (t2 <= 1))  # Shape: (M, N)
    
    # Combine results: intersection occurs if the discriminant is non-negative and there is a valid t in [0, 1]
    intersection_mask = has_real_solutions & intersects  # Shape: (M, N)
    
    return intersection_mask