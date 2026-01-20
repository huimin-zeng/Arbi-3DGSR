#!/bin/bash

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

root_dir="data/tandt"
dataset_name=$(basename "$root_dir")

scenes=("train" "truck")


for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"

    if [ "$DO_TRAIN" = true ]; then
        if [ ! -e "output/${dataset_name}/${scene}/x8/point_cloud/iteration_20000/point_cloud.ply" ]; then
        python train_stage1.py -s data/${dataset_name}/${scene}  --eval --scale 8  --mode stage_2   --iterations 20000      --out_dir output/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations 10_000 20_000 30_000 40_000 50_000 60_000  --checkpoint_iterations  10_000 20_000 30_000 40_000 50_000 60_000   --lambda_fea 0.01   
        fi
        

        if [ ! -e "output/${dataset_name}/${scene}/x8/point_cloud/iteration_40000/point_cloud.ply" ]; then
        python train_stage2.py -s data/${dataset_name}/${scene}  --eval --scale 8  --mode random_stage_4   --lrmodel_path output/${dataset_name}/${scene}/x8/point_cloud/iteration_20000/point_cloud.ply    --iterations 40000  --start_checkpoint  output/${dataset_name}/${scene}/x8/chkpnt20000.pth    --out_dir output/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations  10_000 20_000 30_000 40_000 50_000 60_000 --checkpoint_iterations  10_000 20_000 30_000 40_000 50_000 60_000   --lambda_fea 0.01  
        fi

        if [ ! -e "output/${dataset_name}/${scene}/x8/point_cloud/iteration_60000/point_cloud.ply" ]; then
        python train_stage2.py -s data/${dataset_name}/${scene}  --eval --scale 8  --mode random_stage_8   --lrmodel_path output/${dataset_name}/${scene}/x8/point_cloud/iteration_40000/point_cloud.ply    --iterations 60000  --start_checkpoint  output/${dataset_name}/${scene}/x8/chkpnt40000.pth    --out_dir output/${dataset_name}      --mip      --orthogonal --use_2d reverse_log   --kernel_size 0.2    --save_iterations  10_000 20_000 30_000 40_000 50_000 60_000  --checkpoint_iterations  10_000 20_000 30_000 40_000 50_000 60_000  --lambda_fea 0.01  
        fi
    fi

    if [ "$DO_RENDER" = true ]; then
        python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 1 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 2 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 4 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        # python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 8_3.5 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
        # python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 8_5.7 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2
    #     python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 2_3 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2 --no_gt
    #     python render.py -m output/${dataset_name}/${scene}/x8 --eval --skip_train --scale 1_2 --resolution 1 --mip --use_2d reverse_log --kernel_size 0.2 --no_gt
    # fi
done


if [ "$DO_EVAL" = true ]; then
    python ./cal_metric.py --root_dir  output/tandt --iter_dir ours_60000
fi  

