gpu=3

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 0 0 \
    --iterations 20000 \
    --lr 0.0005 \
    --primary_samples 8000 \
    --backup_samples 0 \
    --pos_score 1 \
    --exp_name baseline_20k \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \



# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 0.1 0 \
#     --iterations 100000 \
#     --lr 0.0001 \
#     --primary_samples 40000 \
#     --backup_samples 0 \
#     --pos_score 1 \
#     --exp_name rholoss_0.1_40k \
#     --img_path ../data/IM41340_0124 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \


# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 0.01 0 \
#     --iterations 100000 \
#     --lr 0.0001 \
#     --primary_samples 40000 \
#     --backup_samples 0 \
#     --pos_score 1 \
#     --exp_name rholoss_0.01_40k \
#     --img_path ../data/IM41340_0124 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size