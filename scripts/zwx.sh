export WANDB_API_KEY=b3a9139b4f82902def9e2675e768ce664219c4ab
export WANDB_MODE=offline
gpu=0,1
CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0.001 0 0\
    --iterations 200 \
    --lr 0.002 \
    --exp_name mean_scale_0 \
    --img_path ./data/1213_demo_data_v2/raw1 \
    --codebook_path ./data/codebook.xlsx \
    --cali_loss_type "mean" \