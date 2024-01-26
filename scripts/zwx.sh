export WANDB_API_KEY=b3a9139b4f82902def9e2675e768ce664219c4ab
export WANDB_MODE=offline
gpu=0,1

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 0 0 0.1\
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_mip_0.1 \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score 0 \

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 0 0 0.01\
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_mip_0.01 \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score 0 \

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 0 0 0.001\
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_mip_0.001 \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 0 0 0\
    --pos_score 0 \