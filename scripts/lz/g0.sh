gpu=3

img_path=../data/IM41340_192
exp_name=crop_processed_num4k_it20k

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
    --iterations 20000 \
    --lr 0.002 \
    --primary_samples 4000 \
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


img_path=../data/IM41340_processed
exp_name=full_processed_num8k_it100k

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
    --iterations 100000 \
    --lr 0.002 \
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


img_path=../data/IM41340_raw
exp_name=full_raw_num8k_it100k

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.001 0.001 0 \
    --iterations 100000 \
    --lr 0.002 \
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