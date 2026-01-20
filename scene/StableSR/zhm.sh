sudo docker run --gpus "device=2"  -it -v /home/public/huimin:/home/public/huimin  --shm-size 64g registry.cn-hangzhou.aliyuncs.com/zenghuimin/zhm_docker:zhm-py310-torch21  /bin/bash


python sr_val_ddpm_text_T_vqganfin_oldcanvas.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt CKPT_PATH --vqgan_ckpt VQGANCKPT_PATH --init-img INPUT_PATH --outdir OUT_DIR --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain

pip install pytorch-lightning==1.4.2 torchmetrics==0.6.0  taming-transformers-rom1504 scikit-learn kornia==0.6 open_clip_torch==2.0.2
cd /home/public/huimin/StableSR 

python sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./inputs/test_example/ --outdir ./output/  --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain


python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./inputs/test_example/ --outdir ./output/  --ddpm_steps 200 --dec_w 0.5 --colorfix_type adain


# DDIM w/ negative prompts
python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_768v_000139.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img ./inputs/train/ --outdir ./output/ --ddim_steps 20 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --upscale 2 --seed 42 --n_samples 1 --input_size 72 --tile_overlap 48 --ddim_eta 1.0

  
CUDA_VISIBLE_DEVICES=0 python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/train/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/train/images_upx2_stable_40/ --upscale 2  --ddim_steps 40 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

  python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/train/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/train/images_upx4_stable_40/ --upscale 4  --ddim_steps 40 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

  python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/train/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/train/images_upx8_stable_40/ --upscale 8  --ddim_steps 40 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

 python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/truck/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/truck/images_upx2_stable/ --upscale 2  --ddim_steps 40 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

  python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/truck/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/truck/images_upx4_stable/ --upscale 4  --ddim_steps 20 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

  python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/truck/images_x8/ --outdir /home/public/huimin/tandt_db/tandt/truck/images_upx8_stable/ --upscale 8  --ddim_steps 20 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

 
# python sr_val_ddim_text_T_negativeprompt_canvas_tile.py --config configs/stableSRNew/v2-finetune_text_T_768v.yaml --ckpt stablesr_000117.ckpt --vqgan_ckpt vqgan_cfw_00011.ckpt --init-img /home/public/huimin/tandt_db/tandt/train/images_x8/ --outdir ./output/ --ddim_steps 20 --dec_w 0.0 --colorfix_type wavelet --scale 7.0 --use_negative_prompt --upscale 8 --seed 42 --n_samples 1 --input_size 512 --tile_overlap 48 --ddim_eta 1.0

# encoder
python main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 0 --name debug --scale_lr False

# WFT
python main.py --train --base configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml --gpus GPU_ID, --name NAME --scale_lr False

  




# try ddim
 
 