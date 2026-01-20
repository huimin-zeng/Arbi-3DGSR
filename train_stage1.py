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
from torch.utils.checkpoint import checkpoint_sequential
import torch.utils.checkpoint as cp

import os
import torch
from random import randint
import torch.nn.functional as F
from utils.loss_utils import l1_loss, ssim,l2_loss,CE_loss
from gaussian_renderer import render, render_mip, network_gui
import sys
from scene import Scene, GaussianModel,GaussianModel_mip
from utils.general_utils import safe_state
import uuid
from models.network_swinir import SwinIR as net
from tqdm import tqdm
import wandb
from utils.image_utils import psnr
from train import prepare_output_and_logger,training_report,create_offset_gt
from taming.modules.losses.lpips import LPIPS
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_fea_idx(vq_encoder,x):
    fea = vq_encoder.encoder(x)
    h = vq_encoder.quant_conv(fea)
    quant, emb_loss, info = vq_encoder.quantize(h)
    return fea, quant, emb_loss, info

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,scale=1,output_dir=None,mode="train",sr_version="stable",orthogonal=False):
    pass



def training_mip(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,scale=1,output_dir=None,mode="train",sr_version="stable",orthogonal=False, num_view=0,use_2d=None,no_pixel=False,lambda_fea=1,lambda_lr=1,lambda_hr=1,col_mip=False):
    orthogonal_pro=True if (orthogonal and num_view>0) else False
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,scale,output_dir)
    gaussians = GaussianModel_mip(dataset.sh_degree,with_stablesr=False,with_stablesr_vq=["diff","vq"])
    stable_scale_factor=gaussians.stablesr_vq["diff"].scale_factor
    # del gaussians.stablesr_vq["diff"]
    del gaussians.stablesr_vq["vq"].decoder
    torch.cuda.empty_cache()
    gaussians.stablesr_vq["vq"].eval()                         # never train it
    for p in gaussians.stablesr_vq["vq"].parameters():         # freeze weights
        p.requires_grad_(False)

    gaussians.stablesr_vq["diff"].eval()                         # never train it
    for p in gaussians.stablesr_vq["diff"].parameters():         # freeze weights
        p.requires_grad_(False)


    scene = Scene(dataset, gaussians,scale=scale,mode=mode,sr_version=sr_version,orthogonal=[orthogonal,num_view],col_mip=col_mip)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
 
    gaussians.compute_3D_filter(cameras=trainCameras)
    if use_2d is not None:
        kernel_size_used=[dataset.kernel_size,use_2d] 
    else:
        kernel_size_used = dataset.kernel_size

    viewpoint_stack = None
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
                    net_image = render_mip(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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

        idx=randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(idx)
        # sr_latent = sr_latent_cache[idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        
        render_pkg = render_mip(viewpoint_cam, gaussians, pipe, background, kernel_size=kernel_size_used, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss

        image.requires_grad_(True)
        SR_image = viewpoint_cam.SR_image.cuda()
        if (orthogonal_pro and (not viewpoint_cam.specified)):
            L_per=0
        else:# require grad!
            image_resize=(F.interpolate(image.unsqueeze(0), size=(512,512), mode='bicubic', )*2-1).clamp(-1,1) 
            SR_image_resize=(F.interpolate(SR_image.unsqueeze(0), size=(512,512), mode='bicubic', )*2-1).clamp(-1,1)
            image_latent = (gaussians.stablesr_vq["vq"].encode(image_resize)[0]).sample() * stable_scale_factor

            with torch.no_grad():# prepare condition
                semantic_c = [gaussians.stablesr_vq["diff"].cond_stage_model(['']*image_resize.size(0))]
                ts = torch.full((1,), int(199), device=gaussians.stablesr_vq["vq"].device, dtype=torch.long)
                struct_cond_input = gaussians.stablesr_vq["diff"].structcond_stage_model(image_latent, ts)
                cond_all = {'c_crossattn':semantic_c,'struct_cond':struct_cond_input}
                SR_image_latent = (gaussians.stablesr_vq["vq"].encode(SR_image_resize)[0]).sample() * stable_scale_factor
                sr_noise_pred = gaussians.stablesr_vq["diff"].model(SR_image_latent, ts, **cond_all) 
            
            noise_pred = gaussians.stablesr_vq["diff"].model(image_latent, ts, **cond_all) 

            grad =  gaussians.stablesr_vq["diff"].lvlb_weights[199] * (noise_pred - sr_noise_pred) 
            grad = torch.nan_to_num(grad)
            targets = (image_latent - grad).detach()
            L_per=l2_loss(image_latent.float(), targets)

        if (orthogonal and (not viewpoint_cam.specified)) or no_pixel or (orthogonal_pro and (not viewpoint_cam.specified)):
            loss_HR=0
        else:
            SR_image = viewpoint_cam.SR_image.cuda()
            loss_HR=l2_loss(image, SR_image)


        # -------------- <render,LR>
        gt_image = viewpoint_cam.original_image.cuda()#input LR
        down_scale=int(viewpoint_cam.up_scale)
        image = F.avg_pool2d(image, kernel_size=(down_scale, down_scale))
        Ll1_lr = l2_loss(image, gt_image)
        loss_lr = (1.0 - opt.lambda_dssim) * Ll1_lr + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = lambda_fea*L_per +  lambda_lr*loss_lr + lambda_hr*loss_HR
        
        loss.backward()

        iter_end.record()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, loss, loss, l2_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_mip, (pipe, background, kernel_size_used))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--scale", type=str, default=None)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000, 60_000, 90_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000,  60_000, 90_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--out_dir", type=str, default = None)
    parser.add_argument("--sr_version", type=str, default = "stable")
    parser.add_argument('--orthogonal', action='store_true', default=False)
    parser.add_argument("--num_view", type=int, default =0)
    parser.add_argument('--mip', action='store_true', default=False)
    parser.add_argument('--use_2d', type=str, default = None)
    parser.add_argument('--no_pixel', action='store_true', default=False)
    parser.add_argument('--lambda_fea', type=float, default = 1.0)
    parser.add_argument('--lambda_lr', type=float, default = 1.0)
    parser.add_argument('--lambda_hr', type=float, default = 1.0)
    parser.add_argument("--wandb_run_name", type=str, default = None)
    parser.add_argument('--col_mip', action='store_true', default=False)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.mip:  
        training_mip(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.scale,args.out_dir,args.mode,args.sr_version,args.orthogonal,args.num_view,args.use_2d,args.no_pixel,args.lambda_fea,args.lambda_lr,args.lambda_hr,args.col_mip)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args.scale,args.out_dir,args.mode,args.sr_version,args.orthogonal,args.num_view)


    # All done
    print("\nTraining complete.")
