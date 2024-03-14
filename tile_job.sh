#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch/_.out
#SBATCH -e sbatch/_.err
#SBATCH -o sbatch/%j.out
#SBATCH -e sbatch/%j.err


# module load gcc-8.3.0
module load cuda-12.1
source activate mrna #激活conda环境
# source activate ce #激活conda环境
cd /home/home/ccnt_zq/gzx/code/mRNA/new/RNA_GS_3D

gpu=0

export WANDB_API_KEY=2e95ec93b7f616b70703b6ba3e97d345fa9caa48
export WANDB_MODE=offline

# img_path=../../data/IM41340_tiled/1_36
img_path=../../data/IM41340_tiled/Nrgn_16
# img_path=../data/IM41340_tiled/0_36

# exp_name=op45_pr0.03_m0_noop
# exp_name=debug_pr0.05_rcla_osig
# exp_name=A1_lr0.0005_pr0.55
# exp_name=Now_lr0.002_0cd0.1_nopr
# exp_name=Nrgn/0.01
# exp_name=Now_lr0.002_pr0.4
# exp_name=Now_lr0.002_pr0.2_times
exp_name=Z_D/cd0.001_p20000

# exp_name=bg0_sz1.53_l80_op0.9_ssim40



# CUDA_VISIBLE_DEVICES=$gpu python /home/home/ccnt_zq/gzx/code/mRNA/new/RNA_GS_3D/train.py \
#     --weights 1.0 0.0 0 0 0 0 0 0 0 0 0 0 \
#     --save-imgs \
#     --iterations 20000 \
#     --lr 0.01 \
#     --primary_samples 20000 \
#     --densification_interval 1000 \
#     --no-initialization \
#     --pos_score 1 \
#     --backup_samples 0 \
#     --exp_name ${exp_name} \
#     --img_path ${img_path} \
#     --codebook_path ../../configs/codebook0.xlsx \
#     --size_range  3 7 \
#     --dens_flags 0 0 0 \
#     --thresholds 0 0.01 6 \

# CUDA_VISIBLE_DEVICES=$gpu python /home/home/ccnt_zq/gzx/code/mRNA/new/RNA_GS_3D/train.py \
#     --weights 0.0 1.0 0 0 0 0 0.001 0 0.1 0.1 0 0 \
#     --save-imgs \
#     --iterations 20000 \
#     --lr 0.002 \
#     --primary_samples 20000 \
#     --densification_interval 1000 \
#     --no-initialization \
#     --pos_score 1 \
#     --backup_samples 0 \
#     --exp_name ${exp_name} \
#     --img_path ${img_path} \
#     --codebook_path ../../configs/codebook0.xlsx \
#     --size_range  3 7 \
#     --dens_flags 1 0 0 \
#     --thresholds 0.02 0.01 6 \

CUDA_VISIBLE_DEVICES=$gpu python /home/home/ccnt_zq/gzx/code/mRNA/new/RNA_GS_3D/train.py \
    --weights 0.0 1.0 0 0 0 0 0 0 0.1 0.1 0 0 \
    --save-imgs \
    --iterations 20000 \
    --lr 0.002 \
    --primary_samples 40 \
    --densification_interval 1000 \
    --no-initialization \
    --pos_score 1 \
    --backup_samples 0 \
    --exp_name ${exp_name} \
    --img_path ${img_path} \
    --codebook_path ../../configs/codebook0.xlsx \
    --size_range  3 7 \
    --dens_flags 1 0 0 \
    --thresholds 0.02 0.01 6 \

python postprocess.py \
    --csv_path outputs/${exp_name}/output_all.csv \
    --img_path ${img_path} \
    --pos_threshold 20

    # --weights 0 1.0 0 0 0 0 0.1 0 0.1 0.1 0 0 \




