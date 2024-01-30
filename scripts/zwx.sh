export WANDB_API_KEY=b3a9139b4f82902def9e2675e768ce664219c4ab
export WANDB_MODE=offline
gpu=0

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 0 0 0\
    --iterations 20000 \
    --lr 0.0015 \
    --exp_name num_8k \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score 0 \
    --primary_samples 8000 \
    --backup_samples 8000 \

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0.001 0 0 0 0 0\
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name num_2w \
#     --img_path ./data/IM41340regi \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score 0 \
#     --primary_samples 20000 \
#     --backup_samples 20000 \

# CUDA_VISIBLE_DEVICES=$gpu python train.py \
#     --weights 0 1 0 0 0 0 0.001 0 0 0 0 0\
#     --iterations 20000 \
#     --lr 0.002 \
#     --exp_name num_3w \
#     --img_path ./data/IM41340regi \
#     --codebook_path ./data/codebook.xlsx \
#     --cali_loss_type "cos" \
#     --dens_flags 0 0 0\
#     --pos_score 0 \
#     --primary_samples 30000 \
#     --backup_samples 30000 \