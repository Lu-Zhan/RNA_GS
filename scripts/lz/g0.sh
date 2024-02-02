gpu=3

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size, rho, mdp, mip
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 0.01 0 0\
    --iterations 20000 \
    --lr 0.0015 \
    --exp_name num_8k_cos_rho \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score 0 \
    --primary_samples 8000 \
    --backup_samples 8000 \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size