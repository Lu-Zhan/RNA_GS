gpu=0

CUDA_VISIBLE_DEVICES=$gpu python train.py \
    --weights 0 1 0 0 0 0 0 0 0 \
    --iterations 200 \
    --lr 0.002 \
    --exp_name debug_test_ckpt \
    --img_path ../data/1213_demo_data_v2/raw1 \
    --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \

# l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size