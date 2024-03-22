#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -o sbatch/_.out
#SBATCH -e sbatch/_.err
#SBATCH -o sbatch/%j.out
#SBATCH -e sbatch/%j.err



module load cuda-12.1
source activate mrna #激活conda环境
cd /home/home/ccnt_zq/gzx/code/mRNA/new/RNA_GS_0315

gpu=0

export WANDB_API_KEY=2e95ec93b7f616b70703b6ba3e97d345fa9caa48
export WANDB_MODE=offline

img_path=../../data/IM41340_tiled/1_36
codebook_path=../../configs/codebook.xlsx
exp_name=l2


python train_2d.py \
        --exp_name ${exp_name} \
        --config ./configs/default_gzx.yaml \