gpu=3

img_path=../data/IM41340_processed
exp_name=full_8k

 v=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
    --iterations 20000 \
    --lr 0.02 \
    --primary_samples 8000 \
    --backup_samples 0 \
    --pos_score 1 \
    --exp_name ${exp_name} \
    --img_path ${img_path} \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --size_range 3 7

python postprocess.py \
    --csv_path outputs/${exp_name}/output_all.csv \
    --img_path ${img_path} \
    --pos_threshold 20


# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 0 0 \
#     --iterations 50000 \
#     --lr 0.005 \
#     --primary_samples 10000 \
#     --backup_samples 0 \
#     --pos_score 1 \
#     --exp_name baseline_40k \
#     --img_path ../data/IM41340_0124 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \



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