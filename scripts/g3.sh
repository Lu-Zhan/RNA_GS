gpu=3


CUDA_VISIBLE_DEVICES=$gpu python train.py --weights 0 1 0 0 0 0 0.001 --iterations 2000 --lr 0.002 --exp_name codecos_0.001
