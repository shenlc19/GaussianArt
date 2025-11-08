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

import os
import torch
from torch import nn
from random import randint
from utils.loss_utils import *
from utils.general_utils import build_rotation, convert_cam_name
from utils.articulation_utils import *
from utils.transform_utils import *
import open3d as o3d
from utils.system_utils import mkdir_p
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from matplotlib.image import imsave
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, num_parts, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, freeze_parts):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, num_parts=num_parts)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # setup articulation params
    N = num_parts
    art_R = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cuda").unsqueeze(0).expand([N, 4]).clone().requires_grad_(True)
    art_T = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda").unsqueeze(0).expand([N, 3]).clone().requires_grad_(True)
    art_T = nn.Parameter(art_T)
    l = [
        {'params': [art_R], 'lr': 0.008, "name": "art_R"},
        {'params': [art_T], 'lr': 0.00016, "name": "art_T"},
    ]
    # l = [
    #     {'params': [art_R], 'lr': 0.02, "name": "art_R"},
    #     {'params': [art_T], 'lr': 0.00008, "name": "art_T"},
    # ]

    art_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    mix_iter = 30000
    # mix_iter = 15000
    # mix_iter = 6000

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    

    
    # for cam in scene.getTrainCameras():
    #     print("相机属性：", dir(cam))
    #     break  # 只看第一个就够了
    cam_dict = {convert_cam_name(cam.image_name, cam.time): cam for cam in (scene.getTrainCameras())}

    # breakpoint()
    # filter_all_corr(cam_dict)
    # TODO: cluster init
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        if iteration < mix_iter:
            # while 'start' not in viewpoint_cam.image_name:
            #     if not viewpoint_stack:
            #         viewpoint_stack = scene.getTrainCameras().copy()
            #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # elif iteration >= mix_iter:
            # while 'end' not in viewpoint_cam.image_name:
            #     if not viewpoint_stack:
            #         viewpoint_stack = scene.getTrainCameras().copy()
            #     viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            while viewpoint_cam.time != 0.0:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image = viewpoint_cam.original_image.cuda()
        if torch.all(gt_image == 0.0):
            continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        detached_R = art_R.clone().detach()
        mask_R = torch.zeros_like(art_R).cuda()

        for idx in freeze_parts:
            mask_R[idx, :] = 1
        # print("mask_R内容", mask_R)
        
        art_R_1 = art_R * (1 - mask_R) + detached_R * mask_R
        # print("art_R_1内容", art_R_1)
        
        #import pdb; pdb.set_trace()

        articulation_matrix = build_rotation(art_R_1)
        articulation_trans = art_T
        
        if iteration > opt.hard_training_step:
            articulation_weights = gaussians.get_weight
            # 获取每行的最大索引
            max_indices = torch.argmax(articulation_weights, dim=1)
            
            # 创建一个形状为 (N, M) 的全零张量
            new_articulation_weights = torch.zeros_like(articulation_weights)
            
            # 将最大索引位置的值设为 1
            new_articulation_weights[torch.arange(articulation_weights.shape[0]), max_indices] = 1
            articulation_weights = new_articulation_weights
        else:
            articulation_weights = gaussians.get_weight
                                           
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                            articulation_weights=articulation_weights, articulation_matrix=articulation_matrix, articulation_trans=articulation_trans)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth, rend_alpha = render_pkg["depth"], render_pkg["depth"]   
        
        # Loss
        Ll1 = l1_loss(image, gt_image)
        Ll1_depth = 0.
        gt_depth = viewpoint_cam.depth.cuda()
        # gt_image = torch.where(torch.abs(gt_image) < 1e-6, torch.tensor(0, dtype=gt_image.dtype).cuda(), gt_image)
        # mask = torch.any(gt_image, dim=0)
        # gt_depth = gt_depth * mask
        depth = depth.squeeze()
        Ll1_depth = l1_loss(depth, gt_depth)
        if iteration < opt.semantic_until_iter:
            gt_semantic = viewpoint_cam.semantic.cuda().unsqueeze(0).long()
            if torch.all(gt_semantic == -1):
                semantic_loss = torch.zeros_like(Ll1)
            else:
                semantic = render_pkg["weight"].unsqueeze(0) # [1, S, H, W]
                semantic_loss = torch.nn.functional.cross_entropy(
                    input=semantic, 
                    target=gt_semantic,
                    ignore_index=-1, 
                    reduction='mean'
                )
                # color_map = {
                #     0: [0, 0, 0],
                #     1: [232, 126, 23],
                #     2: [58, 228, 24],
                #     3: [55, 18, 238],
                #     4: [255, 255, 0],
                #     5: [165, 42, 42],
                #     6: [238, 130, 238],
                #     7: [255, 248, 220],
                # }
                # semantic = semantic.squeeze().detach().cpu().numpy()   
                # N, H, W = semantic.shape

                # # 初始化 result 数组
                # result = np.zeros((N, H, W), dtype=np.uint8)

                # # 对于每个类，计算其概率是否为最大值，并排除全为0的情况
                # max_prob = np.max(semantic, axis=0)
                # for i in range(N):
                #     result[i, :, :] = (semantic[i, :, :] == max_prob).astype(np.uint8)

                # # 创建背景通道，将全为0的区域置为1
                # background_channel = np.zeros((H, W), dtype=np.uint8)
                # background_channel = np.all(semantic == semantic[0, :, :], axis=0)

                # # 将背景通道与 result 数组堆叠在一起
                # result = np.vstack((background_channel[np.newaxis, :, :], result))

                # # 使用 np.argmax 计算每个像素的标签
                # labels = np.argmax(result, axis=0)

                # # 将类别标签映射到对应的颜色
                # rgb_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
                # for label, color in color_map.items():
                #     rgb_image[labels == label] = color
                # imsave(viewpoint_cam.image_name + ".png", rgb_image)
        else:
            semantic_loss = torch.zeros_like(Ll1)
        
               
        # if iteration >= mix_iter:
        #     loss = 0.01 *((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        # else:
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss += 0.5 * Ll1_depth
        loss += 0.5 * Ll1_depth
        # loss += 0.5 * semantic_loss
        loss += 0.5 * semantic_loss
        # loss += 0. * semantic_loss
        
        if opt.lambda_occ > 0:
            mask = torch.any(gt_image, dim=0)
            o= rend_alpha.clamp(1e-6, 1-1e-6)
            mask = (~mask).float()
            loss_mask_opa = (-mask * torch.log(1 - o)).mean()
            loss_opacity_entropy = -(o*torch.log(o)).mean()
            loss += opt.lambda_occ * loss_mask_opa + opt.lambda_opacity_entropy * loss_opacity_entropy
        
        if opt.lambda_scale_flatten > 0:
            scale_flatten_loss = gaussians.scale_flatten_loss()
            loss += opt.lambda_scale_flatten * scale_flatten_loss
        else:
            scale_flatten_loss = None
    
        if iteration > 150000:
            semantic = semantic.squeeze()   
            N, H, W = semantic.shape
            result = torch.zeros((N, H, W), dtype=torch.uint8, device="cuda")
            max_prob = torch.max(semantic, dim=0).values
            for i in range(N):
                result[i, :, :] = (semantic[i, :, :] == max_prob)
            eq = torch.all(result == result[0], dim=0)
            result[:,eq>0]=0
            result = result.unsqueeze(0)
            
            corrloss = corr_loss(cam_dict, viewpoint_cam.correspondence, viewpoint_cam.image_name, result,
                                    articulation_matrix=articulation_matrix, articulation_trans=articulation_trans)
            # loss += 0.25 * corrloss
            loss += 10. * corrloss
        else:
            corrloss = None
           
        if iteration >= 30000 and iteration <= opt.hard_training_step:
        # if iteration >= 16000 and iteration <= opt.hard_training_step:
            knnloss = knn_loss(gaussians, articulation_weights) # knn loss
            loss += 1. * knnloss
            # loss += 0.0 * knnloss
        else:
            knnloss = None

        with torch.autograd.set_detect_anomaly(True):
            loss.backward(retain_graph=True)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Ll1_depth, loss, semantic_loss, corrloss, knnloss, scale_flatten_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                mkdir_p(os.path.join(scene.model_path, "ckpts"))
                torch.save({
                    "articulation_params":{
                        "art_R": art_R.clone().detach().cpu(),
                        "art_T": art_T.clone().detach().cpu(),
                    },
                    #"encoder_state_dict" : encoder.state_dict(),
                }, os.path.join(scene.model_path, "ckpts/ours_{}.pth".format(iteration)))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration >= mix_iter:
                    art_optimizer.step()
                    art_optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Ll1_depth, loss, semantic_loss, corr_loss, knn_loss, scale_flatten_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/semantic_loss', semantic_loss.item(), iteration)
        if Ll1_depth is not None and Ll1_depth !=0 :
            tb_writer.add_scalar('train_loss_patches/depth_loss', Ll1_depth.item(), iteration)
        if corr_loss is not None and corr_loss !=0 :
            tb_writer.add_scalar('train_loss_patches/corr_loss', corr_loss.item(), iteration)
        if knn_loss is not None and knn_loss !=0 :
            tb_writer.add_scalar('train_loss_patches/knn_loss', knn_loss.item(), iteration)
        if scale_flatten_loss is not None and scale_flatten_loss !=0 :
            tb_writer.add_scalar('train_loss_patches/scale_flatten_loss', scale_flatten_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)


    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40000, 45000, 50000, 55000, 60000, 65000, 70000, 71000, 72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 80000, 81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000, 101000, 102000, 103000, 104000, 105000, 106000, 107000, 108000, 109000, 110000, 111000, 112000, 113000, 114000, 115000, 116000, 117000, 118000, 119000, 120000, 121000, 122000, 123000, 124000, 125000, 126000, 127000, 128000, 129000, 130000, 131000, 132000, 133000, 134000, 135000, 136000, 137000, 138000, 139000, 140000, 141000, 142000, 143000, 144000, 145000, 146000, 147000, 148000, 149000, 150000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 15000, 20000, 25000, 30000, 31000, 32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000, 42000, 43000, 44000, 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 69000, 70000, 71000, 72000, 73000, 74000, 75000, 76000, 77000, 78000, 79000, 80000, 81000, 82000, 83000, 84000, 85000, 86000, 87000, 88000, 89000, 90000, 91000, 92000, 93000, 94000, 95000, 96000, 97000, 98000, 99000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_parts", type=int, default=2)
    parser.add_argument("--freeze_parts", nargs="+", type=int, default=[])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.num_parts, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.freeze_parts)

    # All done
    print("\nTraining complete.")