gpu=1



# exp_name=crop400_processed_num4k_it20k

for n in {10..14}
do  
    img_path=../data/IM41340_processed/${n}_36
	exp_name=v2_c1e-3_mdp1_lml2-0.5_bg1_size0.01-3-7_coslr0.1_tile${n}-36

    CUDA_VISIBLE_DEVICES=$gpu python train.py \
        --weights 0 1 0 0.5 1 0 0.001 0 0.01 0 1 \
        --iterations 10000 \
        --lr 0.01 \
        --primary_samples 12000 \
        --backup_samples 0 \
        --pos_score 1 \
        --exp_name ${exp_name} \
        --img_path ${img_path} \
        --codebook_path ../data/1213_demo_data_v2/codebook.xlsx \
        --size_range 3 7

    python postprocess.py \
        --csv_path outputs/${exp_name}/output_all.csv \
        --img_path ${img_path} \
        --pos_threshold 20
done

