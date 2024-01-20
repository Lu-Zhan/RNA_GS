gpu=3

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0.001 0 \
    --iterations 20000 \
    --lr 0.002 \
    --exp_name ablation_circle_loss_0.001 \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size