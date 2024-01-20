export WANDB_API_KEY=b3a9139b4f82902def9e2675e768ce664219c4ab
export WANDB_MODE=offline
gpu=0,1

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_baseline \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score \


# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_cos_loss_prune \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 1 0 0\
#     --pos_score \

# 2
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_cos_loss_0.001_desif \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 1 1\
#     --pos_score \

# 3
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0.001 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_normal_loss_0.001 \
#     --img_path ../data/1213_demo_data_v2/raw1 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
#     --cali_loss_type "normal" \
#     --dens_flags 0 0 0\
# # 4
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0.001 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_mean_loss_0.001 \
#     --img_path ../data/1213_demo_data_v2/raw1 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
#     --cali_loss_type "mean" \
#     --dens_flags 0 0 0\
# # 5
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0.001 0 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_median_loss_0.001 \
#     --img_path ../data/1213_demo_data_v2/raw1 \
#     --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
#     --cali_loss_type "median" \
#     --dens_flags 0 0 0\
# # # 6
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0.01 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_circle_loss_0.01 \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score \


# # 7
# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 0 0.01 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_size_loss_0.01 \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score \

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0 1 0 \
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name ablation_circle_loss_1 \
#     --img_path ./data/1213_demo_data_v2/raw1 \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score \


# 7
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0.1 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_size_loss_0.1_minmax_9.7_10.7 \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score \