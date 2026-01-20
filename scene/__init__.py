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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_mip import GaussianModel_mip
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    # gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=True, resolution_scales=[1.0],no_gt=False,scale='1',mode="train",sr_version="realsrarbi",up_scale_test=1,orthogonal=[False,0],col_mip=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, no_gt=no_gt,mode=mode,sr_version=sr_version,up_scale_test=up_scale_test,orthogonal=orthogonal)
        elif os.path.exists(os.path.join(args.source_path, "sparse")) and col_mip:
            scene_info = sceneLoadTypeCallbacks["Colmap_mip"](args.source_path, args.images, args.eval, no_gt=no_gt,mode=mode,sr_version=sr_version,up_scale_test=up_scale_test,orthogonal=orthogonal)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, no_gt=no_gt,scale=scale,mode=mode,sr_version=sr_version,up_scale_test=up_scale_test,orthogonal=orthogonal)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = [] if mode=="test" else cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mode) 
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, "test")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        #     import copy
        #     import torch
        #     gaussiansv1=copy.deepcopy(self.gaussians)
        #     # gaussiansv1.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_60000","point_cloud.ply"))
        #     gaussiansv1.load_ply("/home/public/huimin/3DGS-ASR/output/output_l2_randpro_mip_2d_log_pix_fea_orth/synthetic_nerf_blender/drums/x8/point_cloud/iteration_90000/point_cloud.ply")
        #     # breakpoint()
        #     gaussiansv1._xyz[:,0]=-gaussiansv1._xyz[:,0]
        #     gaussiansv1._xyz[:,1]=-gaussiansv1._xyz[:,1]
        #     gaussiansv1._xyz[:,2]=-gaussiansv1._xyz[:,2]
        
        #     # gaussiansv1._xyz[:,0]+=1
        #     # gaussiansv1._xyz[:,1]+=1
        #     for v1, now in zip(gaussiansv1.__dict__, self.gaussians.__dict__):
        #         if v1!=now:
        #             print(v1)
        #         if isinstance(getattr(gaussiansv1, v1),torch.Tensor):
        #             # if getattr(gaussiansv1, v1).shape!=getattr(self.gaussians, now).shape:
        #             cat_attr=torch.cat([getattr(gaussiansv1, v1),getattr(self.gaussians, now)],dim=0)
        #             setattr(self.gaussians, v1, cat_attr)
        #         else:
        #             if getattr(gaussiansv1, v1)!=getattr(self.gaussians, now):
        #                 print(v1)
        #     # breakpoint()

        # # gaussiansv1._xyz = torch.empty(0)
        # # self.gaussians._features_dc = torch.empty(0)
        # # self.gaussians._features_rest = torch.empty(0)
        # # self.gaussians._scaling = torch.empty(0)
        # # self.gaussians._rotation = torch.empty(0)
        # # self.gaussians._opacity = torch.empty(0)
        # # self.gaussians.filter_3D

        # # gaussiansv1._xyz = torch.empty(0)
        # # gaussiansv1._features_dc = torch.empty(0)
        # # gaussiansv1._features_rest = torch.empty(0)
        # # gaussiansv1._scaling = torch.empty(0)
        # # gaussiansv1._rotation = torch.empty(0)
        # # gaussiansv1._opacity = torch.empty(0)
        # # gaussiansv1.filter_3D

        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]