gpu=0
# 1
export WANDB_API_KEY=2e95ec93b7f616b70703b6ba3e97d345fa9caa48
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0 \
    --iterations 200 \
    --lr 0.002 \
    --exp_name debug \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
    --cali_loss_type "cos" \
    --dens_flags 1 0 0\
    --no-pos_score \