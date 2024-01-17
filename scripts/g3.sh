gpu=3
export WANDB_API_KEY=29bc31455abf8420a0b09ce5289cf40cbe2ee4ac
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=$gpu python train.py --weights 0 1 0 0 0 0 0.001 --iterations 2000 --lr 0.002 --exp_name codecos_0.001
