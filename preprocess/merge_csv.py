# merge csv files, and save to new file, with new column names

import pandas as pd
import os

base_path = '/media/mldadmin/home/s122mdg39_05/Projects_mrna/rna_gs/outputs'
files = [os.path.join(base_path, f'v2_c1e-3_mdp1_lml2-0.5_bg1_size0.01-3-7_coslr0.1_tile{x}-36', 'output_all_post20.0.csv') for x in range(36)]

tile_size = 2304 / 6

for idx, file in enumerate(files):
    df = pd.read_csv(file)
    df = df.iloc[:, :]

    org_x = df['x']
    org_y = df['y']
    class_name = df['Class name']
    class_index = df['Class index']
    scores = df['cos_simi']

    i = idx // 6
    j = idx % 6

    bias_x = tile_size * j
    bias_y = tile_size * i
    
    bias_xy = pd.DataFrame({
        'x': org_x + bias_x, 
        'y': org_y + bias_y, 
        'class': class_name, 
        'index': class_index, 
        'score': scores
    })

    if idx == 0:
        df = bias_xy
    else:
        df = pd.concat([df, bias_xy], axis=0)

df.to_csv(os.path.join(base_path, 'output_2304.csv'), index=False)

