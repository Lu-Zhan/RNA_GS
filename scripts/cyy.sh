gpu=1

export WANDB_API_KEY=29bc31455abf8420a0b09ce5289cf40cbe2ee4ac
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 \
    --iterations 1000 \
    --lr 0.002 \
    --exp_name ablation_baseline \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --thresholds 0.01 0.03 0.03 \
    --cali_loss_type cos \
    --dens_flags 0 0 0\
    --pos_score 1\