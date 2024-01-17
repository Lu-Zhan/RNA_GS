gpu=0
# 1
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_cos_loss_0.001_prune \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 1 0 0\
# 2
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_cos_loss_0.001_desif \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 1 1\
# 3
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_normal_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "normal" \
    --dens_flags 0 0 0\
# 4
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_mean_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "mean" \
    --dens_flags 0 0 0\
# 5
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_median_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "median" \
    --dens_flags 0 0 0\
# 6
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0.001 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_circle_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
# 7
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.001 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_size_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\