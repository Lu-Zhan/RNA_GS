gpu=3


CUDA_VISIBLE_DEVICES=$gpu python test_res.py --weights 0 1 0 0 0 0 0.001 --iterations 20000 --lr 0.002 --exp_name codecos_0.001_vis_new
