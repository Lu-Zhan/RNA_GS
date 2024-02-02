gpu=1
export WANDB_API_KEY=2e95ec93b7f616b70703b6ba3e97d345fa9caa48
export WANDB_MODE=online
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
