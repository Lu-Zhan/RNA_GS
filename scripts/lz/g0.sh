gpu=0

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --primary_samples 4000 \
    --backup_samples 0 \
    --pos_score 1 \
    --exp_name baseline_4000 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size