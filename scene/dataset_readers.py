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
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import random

import numpy as np

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def find_orthogonal_views(sequence, selected_idx):
    # sequence[selected_idx].specified=True
    selected_pose=sequence[selected_idx]
    # Extract the rotation matrix from the selected pose
    R_selected = selected_pose.R[:3, :3]

    # Define the target orthogonal directions based on the selected pose
    targets = {
        "forward": normalize(R_selected[:, 2]),  # Forward direction
        "neg_forward": normalize(-R_selected[:, 2]),  # Negative forward direction
        "right": normalize(R_selected[:, 0]),  # Right direction
        "left": normalize(-R_selected[:, 0])   # Left direction
    }

    # Initialize results
    closest_views = {key: None for key in targets}
    max_cos_theta = {key: -float('inf') for key in targets}

    # Loop through each pose in the sequence
    for idx,cam in enumerate(sequence):
        pose=cam.R
        R_i = pose[:3, :3]
        f_i = normalize(R_i[:, 2])  # Forward direction of the current pose

        # Compare with each target direction
        for key, target in targets.items():
            cos_theta = np.dot(f_i, target)
            if cos_theta > max_cos_theta[key]:
                max_cos_theta[key] = cos_theta
                closest_views[key] = idx
                

    # Verify orthogonality among the selected views
    orthogonal_views = [closest_views[key] for key in ["forward", "neg_forward", "right", "left"]]
    for idx in orthogonal_views:
        # breakpoint()#image_ori_path
        # sequence[10].image_ori_path
        # sequence[74].image_ori_path
        # sequence[194].image_ori_path
        # sequence[96].image_ori_path
        sequence[idx] = sequence[idx]._replace(specified=True) 

    return sequence
 
 
def find_orthogonal_views_pro(camera_poses, x):
    # Generate x evenly spaced reference directions (on a horizontal plane)
    angles = np.linspace(0, 2 * np.pi, x, endpoint=False)  # x angles around a circle
    reference_dirs = np.array([[np.cos(a), 0, np.sin(a)] for a in angles])
    
    # Initialize dictionary to store results
    orthogonal_views = {i: None for i in range(x)}
    assigned_indices = set()  # Keep track of assigned indices to ensure uniqueness
    
    # Assign the best view for each direction
    for i, ref_dir in enumerate(reference_dirs):
        best_alignment = -np.inf
        best_index = None
        best_pose = None
        
        for index, pose in enumerate(camera_poses):
            if index in assigned_indices:
                continue  # Skip already assigned poses
            
            # Compute viewing direction (normalized)
            viewing_dir = pose.R[:, 2]  # Third column of rotation matrix
            viewing_dir = viewing_dir / np.linalg.norm(viewing_dir)
            
            # Compute alignment with the reference direction
            alignment = np.dot(viewing_dir, ref_dir)
            if alignment > best_alignment:
                best_alignment = alignment
                best_index = index
                best_pose = pose
        
        # Assign the best match
        if best_pose:
            orthogonal_views[i] = {"pose": best_pose, "viewing_dir": best_pose.R[:, 2]}
            assigned_indices.add(best_index)
    
    # Fill missing directions by finding the nearest unassigned views
    for i in range(x):
        if orthogonal_views[i] is None:
            nearest_index, nearest_pose = None, None
            min_distance = float("inf")
            
            for index, pose in enumerate(camera_poses):
                if index in assigned_indices:
                    continue  # Skip already assigned poses
                
                viewing_dir = pose.R[:, 2]
                viewing_dir = viewing_dir / np.linalg.norm(viewing_dir)
                distance = np.linalg.norm(viewing_dir - reference_dirs[i])
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_index = index
                    nearest_pose = pose
            
            # Assign the nearest pose to the missing direction
            if nearest_pose:
                orthogonal_views[i] = {"pose": nearest_pose, "viewing_dir": nearest_pose.R[:, 2]}
                assigned_indices.add(nearest_index)
    

    
    for idx in assigned_indices:
        camera_poses[idx] = camera_poses[idx]._replace(specified=True) 
        print("---------",camera_poses[idx].image_path)
    return camera_poses



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_ori: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    up_scale: int
    specified: bool
    image_ori_path: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, mode="train",sr_version="realsrarbi",up_scale_test=1):
    cam_infos = []
    for _, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        break
    cur_ext_name=os.path.basename(extr.name).split(".")[1]
    ext_name="png" if sr_version=="stable" else cur_ext_name
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

        if intr.model == "SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
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
        
        if intr.model=="SIMPLE_RADIAL" and "images_x" in images_folder:
            if "horns" in images_folder or "trex" in images_folder:
                image_path = os.path.join(images_folder, os.path.basename(extr.name).replace('jpg','png'))
            else:
                image_path = os.path.join(images_folder, os.path.basename(extr.name).replace('JPG','png'))
            image_name = os.path.basename(image_path).split(".")[0]
        else:
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            
        image = Image.open(image_path)
        # random choose a HR gt
        if mode=="train":# random choose gt 
            up_scale=random.choice([2,4,8])
            image_ori_path = os.path.join(images_folder.replace("x8",f"upx{up_scale}_{sr_version}"), os.path.basename(extr.name))
            image_ori = Image.open(image_ori_path.replace(cur_ext_name,ext_name))
        elif "stage" in mode: # stage_2/4/8
            if 'random' in mode:   
                limit_scale=int(mode.split("_")[-1])
                scale_list=[]
                scale_lower=1 if 'randomx1' in mode else 2
                while limit_scale>=scale_lower:
                    scale_list.append(limit_scale)
                    limit_scale=limit_scale//2  
                up_scale=random.choice(scale_list)             
            else:# stage_1/2/4/8
                up_scale=int(mode.split("_")[-1])

            if up_scale>1:
                image_ori_path = os.path.join(images_folder.replace("x8",f"upx{up_scale}_{sr_version}"), os.path.basename(extr.name))
                image_ori = Image.open(image_ori_path.replace(cur_ext_name,ext_name))      
            else:
                image_ori_path=image_path
                image_ori=image
        else:# test
            image_ori=image
            up_scale=up_scale_test
            image_ori_path=None
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_ori=image_ori,image_ori_path=image_ori_path,
                              image_path=image_path, image_name=image_name, width=width, height=height,up_scale=up_scale,specified=False)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_mip(cam_extrinsics, cam_intrinsics, images_folder, mode="train",sr_version="realsrarbi",up_scale_test=1):
    cam_infos = []
    for _, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        break
    cur_ext_name=os.path.basename(extr.name).split(".")[1]
    ext_name="png" if sr_version=="stable" else cur_ext_name
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

        
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # random choose a HR gt
        if mode=="train":# random choose gt 
            up_scale=random.choice([2,4,8])
            image_ori_path = os.path.join(images_folder.replace("x8",f"upx{up_scale}_{sr_version}"), os.path.basename(extr.name))
            image_ori = Image.open(image_ori_path.replace(cur_ext_name,ext_name))
        elif "stage" in mode: # stage_2/4/8
            if 'random' in mode:   
                limit_scale=int(mode.split("_")[-1])
                scale_list=[]
                while limit_scale>=2:
                    scale_list.append(limit_scale)
                    limit_scale=limit_scale//2  
                up_scale=random.choice(scale_list)             
            else:# stage_1/2/4/8
                up_scale=float(mode.split("_")[-1])
                if up_scale>1:
                    image_ori_path = os.path.join(images_folder.replace("x8",f"upx{up_scale}_{sr_version}"), os.path.basename(extr.name))
                    image_ori = Image.open(image_ori_path.replace(cur_ext_name,ext_name))      
                else:
                    image_ori_path=image_path
                    image_ori=image     
        else:# test
            image_ori=image
            up_scale=up_scale_test
            image_ori_path=None
        # breakpoint()
        focal_length_x = intr.params[0]
        focal_length_y = intr.params[1]
        FovY = focal2fov(focal_length_y, image.size[0]*up_scale)
        FovX = focal2fov(focal_length_x, image.size[1]*up_scale)

        breakpoint
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_ori=image_ori,image_ori_path=image_ori_path,
                              image_path=image_path, image_name=image_name, width=width, height=height,up_scale=up_scale,specified=False)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
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

def readColmapSceneInfo(path, images, eval, llffhold=8, no_gt=False,mode="train",sr_version="realsrarbi",up_scale_test=1,orthogonal="orthogonal"):
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
    cam_infos_unsorted=[]
    train_cam_infos=[]
    test_cam_infos=[]
    for reading_dir_each in reading_dir:
        cam_infos_unsorted_train = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each), mode="test",sr_version=sr_version,up_scale_test=up_scale_test) if mode=="test" else readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each), mode=mode,sr_version=sr_version,up_scale_test=up_scale_test)
        cam_infos_train = sorted(cam_infos_unsorted_train.copy(), key = lambda x : x.image_name)
        cam_infos_unsorted_test = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each),mode="test",sr_version=sr_version,up_scale_test=up_scale_test)
        cam_infos_test = sorted(cam_infos_unsorted_test.copy(), key = lambda x : x.image_name)

        if eval:
            train_cam_infos.extend([c for idx, c in enumerate(cam_infos_train) if idx % llffhold != 0])
            test_cam_infos.extend([c for idx, c in enumerate(cam_infos_test) if idx % llffhold == 0])
        else:
            train_cam_infos.extend(cam_infos_train)
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if orthogonal[0]:
        if orthogonal[1]>0:# pro: user-specify N views
            train_cam_infos = find_orthogonal_views_pro(train_cam_infos,orthogonal[1])
        else:
            random_idx= random.randint(0, len(train_cam_infos)-1)
            train_cam_infos = find_orthogonal_views(train_cam_infos,random_idx)


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



def readColmapSceneInfo_mip(path, images, eval, llffhold=8, no_gt=False,mode="train",sr_version="realsrarbi",up_scale_test=1,orthogonal="orthogonal"):
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
    cam_infos_unsorted=[]
    train_cam_infos=[]
    test_cam_infos=[]
    for reading_dir_each in reading_dir:
        cam_infos_unsorted_train = readColmapCameras_mip(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each), mode="test",sr_version=sr_version,up_scale_test=up_scale_test) if mode=="test" else readColmapCameras_mip(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each), mode=mode,sr_version=sr_version,up_scale_test=up_scale_test)
        cam_infos_train = sorted(cam_infos_unsorted_train.copy(), key = lambda x : x.image_name)
        cam_infos_unsorted_test = readColmapCameras_mip(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir_each),mode="test",sr_version=sr_version,up_scale_test=up_scale_test)
        cam_infos_test = sorted(cam_infos_unsorted_test.copy(), key = lambda x : x.image_name)

        if eval:
            train_cam_infos.extend([c for idx, c in enumerate(cam_infos_train) if idx % llffhold != 0])
            test_cam_infos.extend([c for idx, c in enumerate(cam_infos_test) if idx % llffhold == 0])
        else:
            train_cam_infos.extend(cam_infos_train)
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if orthogonal[0]:
        if orthogonal[1]>0:# pro: user-specify N views
            train_cam_infos = find_orthogonal_views_pro(train_cam_infos,orthogonal[1])
        else:
            random_idx= random.randint(0, len(train_cam_infos))
            train_cam_infos = find_orthogonal_views(train_cam_infos,random_idx)


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


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",scale=1, mode="train",sr_version="realsrarbi",up_scale_test=1):
    train_or_test=transformsfile.split("_")[1].split(".json")[0]
    tar_dir=train_or_test if scale=='1' else f"{train_or_test}_x{scale}"
    cam_infos = []
    ext_name=".png" if sr_version=="stable" else ".jpg"
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"].replace(train_or_test,tar_dir) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
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
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            if mode=="train":# random choose gt 
                up_scale=random.choice([2,4,8])
                image_ori_path =  image_path.replace("x8",f"upx{up_scale}_{sr_version}") 
                image_ori = Image.open(image_ori_path.replace(".jpg",ext_name))    

                im_data_ori = np.array(image_ori.convert("RGBA"))
                bg_ori = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data_ori = im_data_ori / 255.0
                arr_ori = norm_data_ori[:,:,:3] * norm_data_ori[:, :, 3:4] + bg_ori * (1 - norm_data_ori[:, :, 3:4])
                image_ori = Image.fromarray(np.array(arr_ori*255.0, dtype=np.byte), "RGB")
            elif "stage" in mode: # stage_2/4/8
                if 'random' in mode:   
                    limit_scale=int(mode.split("_")[-1])
                    scale_list=[]
                    scale_lower=1 if 'randomx1' in mode else 2
                    while limit_scale>=scale_lower:
                        scale_list.append(limit_scale)
                        limit_scale=limit_scale//2  
                    up_scale=random.choice(scale_list) 
                else:#stage_2/4/8
                    up_scale=int(mode.split("_")[-1])
                
                if up_scale>1:
                    image_ori_path =  image_path.replace("x8",f"upx{up_scale}_{sr_version}") 
                    image_ori = Image.open(image_ori_path.replace(".jpg",ext_name))  
                else:
                    image_ori_path=image_path
                    image_ori=image  

                im_data_ori = np.array(image_ori.convert("RGBA"))
                bg_ori = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                norm_data_ori = im_data_ori / 255.0
                arr_ori = norm_data_ori[:,:,:3] * norm_data_ori[:, :, 3:4] + bg_ori * (1 - norm_data_ori[:, :, 3:4])
                image_ori = Image.fromarray(np.array(arr_ori*255.0, dtype=np.byte), "RGB")

            else:# test
                image_ori=image
                up_scale=up_scale_test
                image_ori_path=None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,image_ori=image_ori, image_ori_path=image_ori_path,
                                         up_scale=up_scale, image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],specified=False))
             
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",no_gt=False,scale='1',mode="train",sr_version="realsrarbi",up_scale_test=1,orthogonal="orthogonal"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,'8',mode,sr_version,up_scale_test) if mode=="test" else readCamerasFromTransforms(path, "transforms_train.json", white_background, extension,scale,mode,sr_version,up_scale_test)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension,scale,mode,sr_version,up_scale_test)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if orthogonal[0]: # [True,N]
        if orthogonal[1]>0:# pro: user-specify N views
            train_cam_infos = find_orthogonal_views_pro(train_cam_infos,orthogonal[1])
        else:# [True,0] get 4 views
            random_idx= random.randint(0, len(train_cam_infos))
            train_cam_infos = find_orthogonal_views(train_cam_infos,random_idx)


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

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_mip": readColmapSceneInfo_mip,
    "Blender" : readNerfSyntheticInfo
}