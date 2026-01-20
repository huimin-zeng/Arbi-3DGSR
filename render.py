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
import sys

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render,render_mip
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel,GaussianModel_mip


def render_set(model_path, name, iteration, views, gaussians, pipeline, background,kernel_size,no_gt=False,mip=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    if not no_gt:
        makedirs(gts_path, exist_ok=True)
    render_func=render_mip if mip else render

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_func(view, gaussians, pipeline, background, kernel_size=kernel_size)["render"]

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        if not no_gt:
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, scales=None, no_gt=False,mip=False,use_2d=None):
    with torch.no_grad():
        gaussians = GaussianModel_mip(dataset.sh_degree) if mip else GaussianModel(dataset.sh_degree)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        input_scale=int(dataset.model_path.split("/")[-1].split("x")[1])# scale: output_scale
        if "_" in scales:
            up,down=scales.split("_")
            up_scale= float(input_scale*float(down)/float(up))
        else:
            up_scale=float(input_scale)/float(scales)  #int(input_scale)//int(scales) 
        if use_2d is not None:
            kernel_size_used=[dataset.kernel_size,use_2d] 
        else:
            kernel_size_used = dataset.kernel_size


        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,no_gt=no_gt,scale=scales,mode="test",up_scale_test=up_scale) 

        if not skip_train:
            render_set(dataset.model_path,  f"train_upx{up_scale}", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,kernel_size_used,no_gt,mip)

        if not skip_test:
            render_set(dataset.model_path, f"test_upx{up_scale}", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,kernel_size_used,no_gt,mip)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no_gt", action="store_true")
    parser.add_argument("--scale", type=str, default='1')
    parser.add_argument('--mip', action='store_true', default=False)
    parser.add_argument('--use_2d', type=str, default=None)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,args.scale, args.no_gt, args.mip,args.use_2d)