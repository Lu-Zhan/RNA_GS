export WANDB_API_KEY=b3a9139b4f82902def9e2675e768ce664219c4ab
export WANDB_MODE=offline
gpu=0
# classes = Snap25 Slc17a7 Gad1 Gad2 plp1 MBP GFAP Aqp4 Rgs5
name="test"
CUDA_VISIBLE_DEVICES=$gpu python vis_single_class.py \
    --exp_name $name \