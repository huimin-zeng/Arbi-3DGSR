import os
import glob
import shutil
import pyiqa
from PIL import Image
import torch
import argparse
import numpy as np
dataset_scene={"MipNeRF360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"], "db": ["drjohnson", "playroom"],"synthetic_nerf_blender":['chair', 'drums' , 'ficus' , 'hotdog' , 'lego' , 'materials', 'mic' ,  'ship'],"tandt":["train","truck"], "nerf_llff_data":['fern',  'flower',  'fortress',  'horns',  'leaves',  'orchids',  'room',  'trex']}



# root_dir="/mnt/nvme1/huimin/gaussian-splatting_out/MipNeRF360"
# root_dir="/mnt/nvme1/huimin/gaussian-splatting_out/db"
# root_dir="/mnt/nvme1/huimin/gaussian-splatting_out/tandt"
# root_dir="/mnt/nvme1/huimin/gaussian-splatting_out/synthetic_nerf_blender"

parser = argparse.ArgumentParser(description="Script for processing root directory")
parser.add_argument('--root_dir', type=str, required=True, help="Path to the root directory")
parser.add_argument('--render_dir', type=str, default="renders", help="Path to the root directory")
parser.add_argument('--only_dir', type=str, default=None, help="Path to the root directory")
parser.add_argument('--iter_dir', type=str, default="ours_30000", help="Path to the root directory")

args = parser.parse_args()
root_dir = args.root_dir
render_dir=args.render_dir
iter_dir=args.iter_dir

dataset_name= root_dir.split("/")[-1]
if "tandt" in dataset_name:
    dataset_name="tandt"
if "db" in dataset_name:
    dataset_name="db"
if "nerf_llff_data" in dataset_name:
    dataset_name="nerf_llff_data"

gt_root_dir=f"/mnt/nvme1/huimin/tandt_db/{dataset_name}" if dataset_name in ["tandt","db"] else f"/mnt/nvme1/huimin/{dataset_name}" 

if args.only_dir:
    dataset_scene[dataset_name]=[args.only_dir]

psnr_metric = pyiqa.create_metric('psnr')
ssim_metric = pyiqa.create_metric('ssim')
lpips_metric = pyiqa.create_metric('lpips')
niqe_metric = pyiqa.create_metric('niqe')

import random
random.seed(0)
fid_metric = pyiqa.create_metric('fid')


def cal_metric(metric_name,pred_cache_paths):
    if metric_name=='psnr':
        metric =psnr_metric
    elif metric_name=='ssim':
        metric =ssim_metric
    elif metric_name=='lpips':
        metric =lpips_metric
 
    scores = []
    # for fn in tqdm(pred_cache_paths, total=len(list(pred_cache_paths))):
    for fn in pred_cache_paths:
        # print(str(fn), str(fn).replace('renders', 'gt'))
        score = metric(str(fn), str(fn).replace(f'{render_dir}', 'gt'))
        scores.append(score)
    # print(f'{metric_name}: {torch.stack(scores).mean()}')
    return scores

def print_list(print_all):
    for item in print_all:
        print(item)
def niqe_cal(metric_name="niqe",pred_cache_paths=None):
    scores = []
    for fn in pred_cache_paths:
        score = niqe_metric(str(fn))
        scores.append(score)
    return scores

def fid_cal(metric_name="fid",fid_gt_cache_path=None, pred_cache_path=None):
    return fid_metric(fid_gt_cache_path, pred_cache_path)
 

 
print_all_psnr=[]

for scale in [2.0,4.0,8.0, 3.5, 5.7]:
    psnr_all=[]
    ssim_all=[]
    lpips_all=[]
    for i in dataset_scene[dataset_name]:
        
        scene_dir=f"{root_dir}/{i}/x8/test_upx{scale}/{iter_dir}/{render_dir}"
        # scene_dir=f"{root_dir}/{i}/test_upx{scale}/{iter_dir}/{render_dir}"
        # breakpoint()
        predict_paths=glob.glob(f'{scene_dir}/*.png')
        psnr=cal_metric('psnr',predict_paths)
        ssim=cal_metric('ssim',predict_paths)
        lpips=cal_metric('lpips',predict_paths)
        psnr_all.extend(psnr)
        ssim_all.extend(ssim)
        lpips_all.extend(lpips)
        # breakpoint()
        # print(predict_paths[0])
        print("PSNR/SSIM/LPIPS,{},{},{},{}".format(i,torch.stack(psnr).mean(),torch.stack(ssim).mean(),torch.stack(lpips).mean()))
        print_all_psnr.append("PSNR/SSIM/LPIPS,{},{},{},{}".format(i,torch.stack(psnr).mean(),torch.stack(ssim).mean(),torch.stack(lpips).mean()))
    # print("Ave PSNR/SSIM/LPIPS,-,{},{},{}".format(torch.stack(psnr_all).mean(), torch.stack(ssim_all).mean(),torch.stack(lpips_all).mean()))
    print_all_psnr.append("Ave PSNR/SSIM/LPIPS,-,{},{},{}".format(torch.stack(psnr_all).mean(), torch.stack(ssim_all).mean(),torch.stack(lpips_all).mean()))
print_list(print_all_psnr)

 
# print_all_niqe=[]
# for scale in [12.0,16.0]:
#     niqe_all=[]
#     for i in dataset_scene[dataset_name]:
#         scene_dir=f"{root_dir}/{i}/x8/test_upx{scale}/{iter_dir}/{render_dir}"
#         # scene_dir=f"{root_dir}/{i}/test_upx{scale}/{iter_dir}/{render_dir}"

#         predict_paths=glob.glob(f'{scene_dir}/*.png')
#         niqe=niqe_cal('niqe',predict_paths)
#         niqe_all.extend(niqe)
#         # print("NIQE,{},{}".format(i,torch.stack(niqe).mean()))
#         print_all_niqe.append("NIQE,{},{}".format(i,torch.stack(niqe).mean()))
#     # breakpoint()
#     # print("Ave NIQE,{}".format(torch.stack(niqe_all).mean()))
#     print_all_niqe.append("Ave NIQE,-,{}".format(torch.stack(niqe_all).mean()))
# print_list(print_all_niqe)

 
print_all_fid=[]
for scale in [2.0, 4.0, 8.0, 3.5, 5.7]:
    niqe_all=[]
    predict_paths_all=[]
    gt_paths_all=[]
    for i in dataset_scene[dataset_name]:
        pred_scene_dir=f"{root_dir}/{i}/x8/test_upx{scale}/{iter_dir}/{render_dir}"
        # pred_scene_dir=f"{root_dir}/{i}/test_upx{scale}/{iter_dir}/{render_dir}"

        if scale<12:
            # breakpoint()
            gt_scene_dir=pred_scene_dir.replace(f"{render_dir}","gt") 
        else:
            gt_scene_dir=f"{root_dir}/{i}/x8/test_upx8.0/{iter_dir}/gt"
            # gt_scene_dir=f"{root_dir}/{i}/test_upx8/{iter_dir}/gt"

        gt_paths_all.extend(glob.glob(f"{gt_scene_dir}/*.png"))
        print("--------",gt_scene_dir,pred_scene_dir)
        fid=fid_cal('fid',gt_scene_dir,pred_scene_dir)
        niqe_all.append(fid)
        # print("FID,{},{}".format(i,fid))
        print_all_fid.append("FID,{},{}".format(i,fid))
        # breakpoint()
        predict_paths_all.extend(glob.glob(f"{pred_scene_dir}/*.png"))
    print_all_fid.append("Ave FID,-,{}".format(np.array(niqe_all).mean()))
    # print("Ave FID,-,{}".format(np.array(niqe_all).mean()))

# print(print_all_fid)
        
#     all_scene_dir_pred=f"{root_dir}/all_test_upx{scale}/{render_dir}"
#     all_scene_dir_gt=f"{root_dir}/all_test_upx{scale}/gt"
#     if not os.path.exists(all_scene_dir_pred):
#         os.makedirs(all_scene_dir_pred)
#     if not os.path.exists(all_scene_dir_gt):
#         os.makedirs(all_scene_dir_gt)
#     # breakpoint()
#     for pred,gt in zip(predict_paths_all,gt_paths_all):
#         scene=pred.split("/")[-6]
#         if not os.path.exists(f"{all_scene_dir_pred}/{scene}_{os.path.basename(pred)}"):
#             os.symlink(pred,f"{all_scene_dir_pred}/{scene}_{os.path.basename(pred)}")
#         if not os.path.exists(f"{all_scene_dir_gt}/{scene}_{os.path.basename(gt)}"):
#             os.symlink(gt,f"{all_scene_dir_gt}/{scene}_{os.path.basename(gt)}")
#     # breakpoint()
#     # print("Ave FID,{}".format(fid_cal('fid',all_scene_dir_gt,all_scene_dir_pred)))
#     print_all_fid.append("Ave FID,-,{}".format(fid_cal('fid',all_scene_dir_gt,all_scene_dir_pred)))
# print_list(print_all_fid)
 
# for niqe in print_all_niqe:
#     print_all_psnr.append(niqe)
 
print_all_psnr_fid=[]
lent=len(dataset_scene[dataset_name])+1
group=len(print_all_psnr)//lent

for i in range(group):
    for psnr,fid in zip(print_all_psnr[i*lent:(i+1)*lent],print_all_fid[i*lent:(i+1)*lent]):
        if "LPIPS" in psnr:
            psnr_fid=f"{psnr.replace('LPIPS','LPIPS/FID')},{fid.split(',')[-1]}"
        else:
            psnr_fid=f"{psnr.replace('NIQE','NIQE/FID')},{fid.split(',')[-1]}"
        print_all_psnr_fid.append(psnr_fid)
print_list(print_all_psnr_fid)

# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/3DGS-ASR/output/output_l2_0001diff_stage4/synthetic_nerf_blender --iter_dir ours_30000 >>/home/zeng.huim/3DGS-ASR/output/output_l2_0001diff_stage4/synthetic_nerf_blender.txt


# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/gaussian-splatting/output/nerf_llff_data  --iter_dir ours_30000 --render_dir  renders_white >>/home/zeng.huim/gaussian-splatting/output/nerf_llff_data/3DGS.txt


# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/gaussian-splatting/output/nerf_llff_data  --iter_dir ours_30000 --render_dir  renders_white_bic >>/home/zeng.huim/gaussian-splatting/output/nerf_llff_data/3DGS_bic.txt

# python /home/zeng.huim/scripts/cal_metric.py --root_dir /scratch/zeng.huim/3DASR_out/output_l2_0001diff/synthetic_nerf_blender --iter_dir ours_30000 >>/scratch/zeng.huim/3DASR_out/output_l2_0001diff/synthetic_nerf_blender.txt
 
# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/mip-splatting/output/MipNeRF360  --iter_dir ours_30000 >>/home/zeng.huim/3DGS-ASR/output_smile/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/MipNeRF360_30k.txt


# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/3DGS-ASR/output_smile/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/synthetic_nerf_blender --iter_dir ours_60000 >>/home/zeng.huim/3DGS-ASR/output_smile/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/synthetic_nerf_blender_60k.txt
# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/3DGS-ASR/output_smile/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/MipNeRF360 --iter_dir ours_90000 >>/home/zeng.huim/3DGS-ASR/output_smile/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/MipNeRF360_90k.txt

# python /home/zeng.huim/scripts/cal_metric.py --root_dir /home/zeng.huim/mip-splatting/output/synthetic_nerf_blender  --iter_dir ours_30000  >>/home/zeng.huim/mip-splatting/output/synthetic_nerf_blender.txt 