gpu=3

export WANDB_API_KEY=2e95ec93b7f616b70703b6ba3e97d345fa9caa48
export WANDB_MODE=offline


CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --primary_samples 4000 \
    --backup_samples 0 \
    --pos_score 1 \
    --exp_name debug2 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../configs/codebook.xlsx \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size, rho
# ../data/IM41340regi_192
# ../data/1213_demo_data_v2/raw1 \
