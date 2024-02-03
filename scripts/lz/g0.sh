gpu=3

img_path=../data/IM41340_processed/400x400
# exp_name=crop400_processed_num4k_it20k
exp_name=baseline_12k_s400_c1e-3_mdp1_rho1e-3

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 0 1 \
    --iterations 20000 \
    --lr 0.003 \
    --primary_samples 12000 \
    --backup_samples 0 \
    --pos_score 1 \
    --exp_name ${exp_name} \
    --img_path ${img_path} \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --size_range 2 4

python postprocess.py \
    --csv_path outputs/${exp_name}/output_all.csv \
    --img_path ${img_path} \
    --pos_threshold 20


# img_path=../data/IM41340_processed
# exp_name=full_processed_num8k_it100k

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
#     --iterations 100000 \
#     --lr 0.002 \
#     --primary_samples 8000 \
#     --backup_samples 0 \
#     --pos_score 1 \
#     --exp_name ${exp_name} \
#     --img_path ${img_path} \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
#     --size_range 3 7

# python postprocess.py \
#     --csv_path outputs/${exp_name}/output_all.csv \
#     --img_path ${img_path} \
#     --pos_threshold 20


# img_path=../data/IM41340_raw
# exp_name=full_raw_num8k_it100k

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
#     --iterations 100000 \
#     --lr 0.002 \
#     --primary_samples 8000 \
#     --backup_samples 0 \
#     --pos_score 1 \
#     --exp_name ${exp_name} \
#     --img_path ${img_path} \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
#     --size_range 3 7

# python postprocess.py \
#     --csv_path outputs/${exp_name}/output_all.csv \
#     --img_path ${img_path} \
#     --pos_threshold 20