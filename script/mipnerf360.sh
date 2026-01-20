#!/bin/bash
# sudo docker run --gpus all -it -v  /mnt/nvme1/huimin:/work/smile/huimin -v /home/public/huimin:/home/public/huimin  --shm-size 64g registry.cn-hangzhou.aliyuncs.com/zenghuimin/zhm_docker:zhm-py310-torch21  /bin/bash
# cd /home/public/huimin/3DGS-ASR 

# pip install pytorch-lightning==1.4.2 torchmetrics==0.6.0  taming-transformers-rom1504 scikit-learn kornia==0.6 open_clip_torch==2.0.2 transformers clip submodules/diff-gaussian-rasterization submodules/simple-knn

# 解析命令行参数
DO_TRAIN=false
DO_RENDER=false
DO_EVAL=false

for arg in "$@"; do
    case $arg in
        --train)
            DO_TRAIN=true
            ;;
        --render)
            DO_RENDER=true
            ;;
        --eval)
            DO_EVAL=true
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--train] [--render] [--eval]"
            exit 1
            ;;
    esac
done

if [ "$DO_TRAIN" = false ] && [ "$DO_RENDER" = false ] && [ "$DO_EVAL" = false ]; then
    DO_TRAIN=true
    DO_RENDER=true
    DO_EVAL=true
fi

root_dir="/work/smile/huimin/MipNeRF360"
dataset_name=$(basename "$root_dir")

scenes=("bicycle" "flowers" "garden" "stump" "treehill" "room" "counter" "kitchen" "bonsai")


for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"

    if [ "$DO_TRAIN" = true ]; then
        if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_30000/point_cloud.ply" ]; then
        python train_stage1.py -s /work/smile/huimin/${dataset_name}/${scene}   --eval --scale 8 --mode stage_2  --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}   --iterations 30000  --mip     --orthogonal --use_2d reverse_log  --kernel_size 0.2   
        fi

        if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_60000/point_cloud.ply" ]; then
        python train_stage2.py -s /work/smile/huimin/${dataset_name}/${scene}    --eval --scale 8  --mode random_stage_4   --lrmodel_path output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_30000/point_cloud.ply    --iterations 60000  --start_checkpoint  output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/chkpnt30000.pth    --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2 
        fi

        if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_90000/point_cloud.ply" ]; then
        python train_stage2.py -s /work/smile/huimin/${dataset_name}/${scene}    --eval --scale 8  --mode random_stage_8  --lrmodel_path output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_60000/point_cloud.ply    --iterations 90000    --start_checkpoint  output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/chkpnt60000.pth    --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}     --mip      --orthogonal --use_2d reverse_log    --kernel_size 0.2 
        fi

        # if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_10000/point_cloud.ply" ]; then
        # python train_vq_vqgrad_sds.py -s /work/smile/huimin/${dataset_name}/${scene}  --eval --scale 8  --mode stage_2   --iterations 10000      --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations 10_000 20_000 30_000 40_000 --checkpoint_iterations 10_000 20_000 30_000 40_000 --port 6019 --lambda_fea 0.1
        # fi

        # if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_20000/point_cloud.ply" ]; then
        # python train_vq_randompro_vqgrad_sds.py -s /work/smile/huimin/${dataset_name}/${scene}  --eval --scale 8  --mode random_stage_4   --lrmodel_path output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_10000/point_cloud.ply    --iterations 20000  --start_checkpoint  output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/chkpnt10000.pth    --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations 10_000 20_000 30_000 40_000 --checkpoint_iterations 10_000 20_000 30_000 40_000 --port 6019 --lambda_fea 0.1
        # fi

        # if [ ! -e "output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_30000/point_cloud.ply" ]; then
        # python train_vq_randompro_vqgrad_sds.py -s /work/smile/huimin/${dataset_name}/${scene}  --eval --scale 8  --mode random_stage_8   --lrmodel_path output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/point_cloud/iteration_20000/point_cloud.ply    --iterations 30000  --start_checkpoint  output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8/chkpnt20000.pth    --out_dir output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations 10_000 20_000 30_000 40_000 --checkpoint_iterations 10_000 20_000 30_000 40_000 --port 6019 --lambda_fea 0.1
        # fi
    fi

    if [ "$DO_RENDER" = true ]; then
        python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 1 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 2 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 4 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        # python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 8_3.5 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        # python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 8_5.7 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        # python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 2_3 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2 --no_gt
        # python render.py -m output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name}/${scene}/x8 --eval --skip_train --scale 1_2 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2 --no_gt
    fi
done

if [ "$DO_EVAL" = true ]; then
    python ./cal_metric.py --root_dir  output/output_l2_randpro_mip_2d_log_pix_fea_orth_norm_abK/k02_reverselog/${dataset_name} --iter_dir ours_90000
fi  
