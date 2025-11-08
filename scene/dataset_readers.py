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
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import cv2 as cv
import glob
from utils.general_utils import *

# class CameraInfo(NamedTuple):
#     uid: int
#     R: np.array
#     T: np.array
#     FovY: np.array
#     FovX: np.array
#     image: np.array
#     image_path: str
#     image_name: str
#     width: int
#     height: int

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    mono_depth: Optional[np.array] = None
    semantic: Optional[np.array] = None

class CameraInfoWithCorr(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    mask: np.array
    semantic: np.array
    correspondence: np.array
    image_path: str
    image_name: str
    width: int
    height: int

# class SceneInfo(NamedTuple):
#     point_cloud: BasicPointCloud
#     train_cameras: list
#     test_cameras: list
#     nerf_normalization: dict
#     ply_path: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_cameras_2s: list
    test_cameras_2s: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.zeros_like(colors)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
#     cam_infos = []

#     with open(os.path.join(path, transformsfile)) as json_file:
#         contents = json.load(json_file)
#         fovx = contents["camera_angle_x"]

#         frames = contents["frames"]
#         for idx, frame in enumerate(frames):
#             cam_name = os.path.join(path, frame["file_path"] + extension)

#             # NeRF 'transform_matrix' is a camera-to-world transform
#             c2w = np.array(frame["transform_matrix"])
#             # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#             c2w[:3, 1:3] *= -1

#             # get the world-to-camera transform and set R, T
#             w2c = np.linalg.inv(c2w)
#             R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#             T = w2c[:3, 3]

#             image_path = os.path.join(path, cam_name)
#             image_name = Path(cam_name).stem
#             image = Image.open(image_path)

#             im_data = np.array(image.convert("RGBA"))

#             bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

#             norm_data = im_data / 255.0
#             arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
#             image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

#             fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
#             FovY = fovy 
#             FovX = fovx
            
#             semantic_path = image_path.replace('_colors', '_instance')
#             semantic = np.array(Image.open(semantic_path)).astype(np.int64)
#             semantic -= 1
#             depth_path = image_path.replace('_colors', '_depth')
#             depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.

#             cam_infos.append(CameraInfoWithCorr(uid=int(id), R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, correspondence=correspondence, mask=mask, semantic=semantic,
#                                 image_path=image_path, image_name=label+str(id), width=image.size[0], height=image.size[1]))
            
#     return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCameraMultiscan(path, transformsfile, white_background, prefix, extension=".jpg", load_depth=False, load_corr=False, load_mask=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as jsonl_file:
        lines = jsonl_file.readlines()
        for idx, line in enumerate(lines):
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(idx+1, len(lines)))
            sys.stdout.flush()

            cam_params = json.loads(line)
            c2w = np.reshape(np.array(cam_params['transform']), [4, 4], order='F')
            # c2w[:3, 3] = c2w[3, :3]
            # c2w[3, :3] = 0.

            c2w[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            image_path = os.path.join(path, 'images', f'frame_{idx:d}'+extension)
            image_name = os.path.basename(image_path).split('.')[0]
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            w, h = image.size
            
            im_data = np.array(image.convert('RGB'))
            
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            
            norm_data = im_data / 255.0
            image = Image.fromarray(np.array(norm_data*255.0, dtype=np.byte), 'RGB')
            
            cam_intrinsics = np.reshape(np.array(cam_params['intrinsics']), [3, 3])
            fovx = 2 * np.arctan(w / (2*cam_intrinsics[0][0]))
            fovy = 2 * np.arctan(h / (2*cam_intrinsics[1][1]))
            
            FovY = fovy 
            FovX = fovx

            depth = np.empty(0)
            if load_depth:
                depth_path = os.path.join(path, 'depth', f'{idx:d}.png')
                if os.path.exists(depth_path):
                    depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.

            mask = np.empty(0)
            if load_mask:
                mask_path = os.path.join(path, 'mask', f'frame_{idx:d}.png')
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.

            correspondence = np.empty(0)
            if load_corr:
                corr_path = os.path.join(path, 'correspondence', f'frame_{idx:d}.npz')
                if os.path.exists(corr_path):
                    correspondence = np.load(corr_path, allow_pickle=True)['data']
                    for idx in range(len(correspondence)):
                        cur_corr = correspondence[idx]
                        for key in list(cur_corr.keys()):
                            new_key = convert_multiscan_dict_name(key)
                            cur_corr[new_key] = cur_corr.pop(key)
            
            cam_infos.append(CameraInfoWithCorr(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, correspondence=correspondence, mask=mask,
                            image_path=image_path, image_name=prefix+image_name, width=image.size[0], height=image.size[1]))

        return cam_infos
    
def readMultiscanInfo(path, white_background, eval, extension=".png"):
    cam_infos = readCameraMultiscan(path, '{}.jsonl'.format(os.path.basename(path)), white_background, extension)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
        pass
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3d.ply')
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

def readParisStyleMultiscanInfo(path, white_background, eval, extension=".png"):
    scene_name = os.path.basename(path)
    start_path = os.path.join(path, scene_name+"_01")
    end_path = os.path.join(path, scene_name+"_00")
    print("Reading Start Cameras")
    start_cam_infos = readCameraMultiscan(start_path, os.path.join(start_path, '{}.jsonl'.format(os.path.basename(start_path))), white_background, "start_", extension, load_depth=True, load_corr=True, load_mask=True)
    print("Reading End Cameras")
    end_cam_infos = readCameraMultiscan(end_path, os.path.join(end_path, '{}.jsonl'.format(os.path.basename(end_path))), white_background, "end_", extension, load_depth=True, load_corr=True, load_mask=True)
    cam_infos = start_cam_infos + end_cam_infos

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]
        pass

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'start.ply')
    pcd = fetchPly(ply_path)
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    return scene_info

def readParisCameras(path, transformsfile, white_background, extension=".png", load_depth=False, load_mask=False, load_corr=False, load_semantic=False, is_debug=False):
    cam_infos = []
    base_path = os.path.dirname(path)
    with open(os.path.join(
        path, transformsfile
        )) as json_file:
        json_data = json.load(json_file)
        cam_intrinsics = np.array(json_data['K'])
        img_base_path = os.path.join(path, transformsfile.split('.')[0].split('_')[-1])
        
        if 'start' in img_base_path.split('/'):
            label = 'start_'
        else:
            label = 'end_'
        
        for id in json_data:
            if id == 'K':
                continue
            
            c2w = np.array(json_data[id], order='F')
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            image_path = os.path.join(img_base_path, id + extension)
            image = Image.open(image_path)
            w, h = image.size
            im_data = np.array(image.convert('RGBA'))
            norm_data = im_data / 255.0
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            arr = norm_data[...,:3] * norm_data[...,-1:] + bg * (1 - norm_data[..., -1:])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), 'RGB')
            
            fovx = 2 * np.arctan(w / (2*cam_intrinsics[0][0]))
            fovy = 2 * np.arctan(h / (2*cam_intrinsics[1][1]))
            
            FovY = fovy 
            FovX = fovx

            depth = np.empty(0)
            if load_depth:
                timestep = {"start_": 0, "end_": 1}[label]
                idx = int(id)
                depth_filename = f"{timestep:05d}_{idx:03d}.png"
                depth_path = os.path.join(base_path, "depth_filtered", depth_filename)
                if os.path.exists(depth_path):
                    depth = np.array(Image.open(depth_path)).astype(np.float32) / 1000.

            mask = np.empty(0)
            if load_mask:
                timestep = {"start_": 0, "end_": 1}[label]
                idx = int(id)
                mask_filename = f"{timestep:05d}_{idx:03d}.png"
                mask_path = os.path.join(base_path, "mask", mask_filename)
                if os.path.exists(mask_path):
                    mask = np.array(Image.open(mask_path)).astype(np.float32)

            correspondence = np.empty(0)
            # need some modification if we use roma
            if load_corr:
                timestep = {"start_": 0, "end_": 1}[label]
                idx = int(id)
                corr_filename = f"src_{timestep:05d}_{idx:03d}_tgt_all_top30.npz"
                corr_path = os.path.join(base_path, "correspondence_loftr/no_filter", corr_filename)
                if os.path.exists(corr_path):
                    correspondence = np.load(corr_path, allow_pickle=True)['data']
            
            semantic = np.empty(0)
            if load_semantic:
                semantic_filename = f"{idx:04d}.npy"
                if label == 'start_':
                    semantic_path = os.path.join(base_path, "start", "semantic", semantic_filename)
                else:
                    semantic_path = os.path.join(base_path, "end", "semantic", semantic_filename)
                if os.path.exists(semantic_path):
                    semantic = np.load(semantic_path)
            
            cam_infos.append(CameraInfoWithCorr(uid=int(id), R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, correspondence=correspondence, mask=mask, semantic=semantic,
                                image_path=image_path, image_name=label+str(id), width=image.size[0], height=image.size[1]))

    return cam_infos

def readParisSceneInfo(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms Start")
    start_train_cam_infos = readParisCameras(os.path.join(path, 'start'), "camera_train.json", white_background, extension, load_depth=True, load_mask=True, load_semantic=True, load_corr=True)
    print("Reading Test Transforms Start")
    start_test_cam_infos = readParisCameras(os.path.join(path, 'start'), "camera_test.json", white_background, extension)
    end_train_cam_infos = []
    end_test_cam_infos = []
    print("Reading Training Transforms End")
    end_train_cam_infos = readParisCameras(os.path.join(path, 'end'), "camera_train.json", white_background, extension, load_depth=True, load_mask=True, load_semantic=True, load_corr=True)
    print("Reading Test Transforms End")
    end_test_cam_infos = readParisCameras(os.path.join(path, 'end'), "camera_test.json", white_background, extension)
    
    train_cam_infos = start_train_cam_infos + end_train_cam_infos
    test_cam_infos = start_test_cam_infos + end_test_cam_infos
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    actual_path = path

    ply_path = os.path.join(actual_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readBlenderStyleParis(path, white_background, eval, extension=".png", ply_path=None):
    print("Reading Training Transforms Start")
    start_train_cam_infos = readCamerasFromTransforms(os.path.join(path, 'start'), "transforms_train.json", white_background, extension, load_depth=True, load_mask=True, load_semantic=True)
    print("Reading Test Transforms Start")
    start_test_cam_infos = readCamerasFromTransforms(os.path.join(path, 'start'), "transforms_test.json", white_background, extension)
    end_train_cam_infos = []
    end_test_cam_infos = []
    print("Reading Training Transforms End")
    end_train_cam_infos = readCamerasFromTransforms(os.path.join(path, 'end'), "transforms_train.json", white_background, extension, load_depth=True, load_mask=True, load_semantic=True)
    print("Reading Test Transforms End")
    end_test_cam_infos = readCamerasFromTransforms(os.path.join(path, 'end'), "transforms_test.json", white_background, extension)
    
    train_cam_infos = start_train_cam_infos + end_train_cam_infos
    test_cam_infos = start_test_cam_infos + end_test_cam_infos
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    actual_path = path

    ply_path = os.path.join(actual_path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readScalableCameras(path, transformsfile, white_background, extension=".png", is_debug=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as cam_log_file:
        c2w_raw_data = []
        cam_id = 0
        for idx, line in enumerate(cam_log_file.readlines()):
            if idx % 7 == 0:
                cam_id = int(line)
            elif idx % 7 == 1:
                fx, fy, cx, cy = [float(z) for z in line.strip('\n').split()]
            elif idx % 7 > 2:
                c2w_raw_data.append([float(z) for z in line.strip('\n').split()])
                pass
            if idx % 7 == 6:
                c2w = np.array(c2w_raw_data, dtype=np.float32)
                c2w[1:3, :3] *= -1
                c2w[3, :3] = c2w[:3, 3]
                c2w[:3, 3] = 0.
                c2w = np.reshape(c2w.reshape(-1).tolist(), [4, 4], order='F')
                # c2w[1:3, :3] *= -1
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1
                c2w_raw_data = []

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                c2w = []
                R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                image_path = os.path.join(path, 'images', f'{idx:d}'+extension)
                image_name = os.path.basename(image_path)
                if not os.path.exists(image_path):
                    continue
                image = Image.open(image_path)
                w, h = image.size
                
                im_data = np.array(image.convert('RGB'))
                norm_data = im_data / 255.0
                image = Image.fromarray(np.array(norm_data*255.0, dtype=np.byte), 'RGB')

                fovx = 2 * np.arctan(w / (2 * fx))
                fovy = 2 * np.arctan(h / (2 * fy))
                
                FovY = fovy 
                FovX = fovx

                cam_infos.append(CameraInfo(uid=cam_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    
    return cam_infos

def readScalableSceneInfo(path, white_background, eval, extension=".png", ply_path=None):
    cam_infos = readScalableCameras(path, "camera.log", white_background, extension)
    train_cam_infos = cam_infos
    test_cam_infos = []

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 8 == 0]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        import trimesh
        mesh = trimesh.load(os.path.join(path, 'mesh.ply'))
        xyz = np.asarray(mesh.vertices).astype(np.float32)
        num_pts = xyz.shape[0]
        shs = np.zeros((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=np.zeros_like(xyz, dtype=np.float32), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, np.zeros_like(xyz, dtype=np.float32))
        pass
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", no_bg=False, load_depth=True, load_mono_depth=True, load_semantic=True):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        fovy = contents["camera_angle_y"]

        frames = contents["frames"]
        # frames = sorted(frames, key=lambda x: int(os.path.basename(x['file_path']).split('.')[0].split('_')[-1]))
        frames = sorted(frames, key=lambda x: x['file_path'])
        for idx, frame in enumerate(frames):
            cam_name = frame["file_path"]
            if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba')):
                cam_name = os.path.join(os.path.dirname(os.path.dirname(os.path.join(path, cam_name))), 'rgba', os.path.basename(cam_name)).replace('.jpg', '.png')
            if cam_name.endswith('jpg') or cam_name.endswith('png'):
                cam_name = os.path.join(path, cam_name)
            else:
                cam_name = os.path.join(path, cam_name + extension)
            frame_time = frame['time']

            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            try:
                im_data = np.array(image.convert("RGBA"))
            except:
                print(f'{image_path} is damaged')
                continue

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] 
            if no_bg:
                norm_data[:, :, :3] = norm_data[:, :, 3:4] * norm_data[:, :, :3] + bg * (1 - norm_data[:, :, 3:4])
            
            arr = np.concatenate([arr, mask], axis=-1)

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGBA" if arr.shape[-1] == 4 else "RGB")

            FovY = fovy
            FovX = fovx

            idx = str(int(image_name)).zfill(3)
            depth_path = image_path.replace('rgba', 'depth')
            if load_depth :
                depth = cv.imread(depth_path, -1) / 1e3
                h, w = depth.shape
                if depth.size == mask.size:
                    depth[mask[..., 0] < 0.5] = 0
                else:
                    depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
                depth[depth < 0.1] = 0
            else:
                depth = None

            mono_depth_path = image_path.replace('rgba', 'mono_depth')
            if load_mono_depth and os.path.exists(mono_depth_path):
                mono_depth = cv.imread(mono_depth_path, cv.IMREAD_GRAYSCALE) / 255
                h, w = mono_depth.shape and os.path.exists(depth_path)
                if mono_depth.size == mask.size:
                    mono_depth[mask[..., 0] < 0.5] = 0
                else:
                    mono_depth[cv.resize(mask[..., 0], [w, h], interpolation=cv.INTER_NEAREST) < 0.5] = 0
            else:
                mono_depth = None
                
            semantic_path = image_path.replace('rgba', 'semantic').replace('png', 'npy')
            if load_semantic and os.path.exists(semantic_path):
                semantic = np.load(semantic_path)
            else:
                semantic = None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mono_depth=mono_depth, semantic=semantic,
                                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], fid=frame_time))

    return cam_infos

def readInfo_2states(path, white_background, eval, extension=".png", no_bg=True):
    print("Reading Training Transforms")
    train_cam_infos = []
    test_cam_infos = []
    for state in ['start', 'end']:
        train_infos = readCamerasFromTransforms(
            path, f"transforms_train_{state}.json", white_background, extension, no_bg=no_bg)
        try:
            test_infos = readCamerasFromTransforms(
                path, f"transforms_test_{state}.json", white_background, extension, no_bg=no_bg)
        except:
            test_infos = []
        if not eval:
            train_infos.extend(test_infos)
        train_cam_infos.append(train_infos)
        test_cam_infos.append(test_infos)
        print(f"Read train_{state} transforms with {len(train_infos)} cameras")
        print(f"Read test_{state} transforms with {len(test_infos)} cameras")

    nerf_normalization = getNerfppNorm(train_cam_infos[0] + train_cam_infos[1])

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos[0] + train_cam_infos[1],
                           test_cameras=test_cam_infos[0] + test_cam_infos[1],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_cameras_2s=train_cam_infos,
                           test_cameras_2s=test_cam_infos)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readInfo_2states,
    "Multiscan": readMultiscanInfo,
    "Paris": readParisSceneInfo,
    "Scalable": readScalableSceneInfo,
    "ParisStyleMultiscan": readParisStyleMultiscanInfo,
    "BlenderStyleParis": readBlenderStyleParis,
}