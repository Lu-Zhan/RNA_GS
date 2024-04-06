import os
import torch

import numpy as np
import pandas as pd
from skimage import filters
import tyro

from pathlib import Path
from typing import Optional
import argparse

import matplotlib.pyplot as plt

from torch.nn.functional import grid_sample


def read_codebook(path, bg=False):
    df = pd.read_excel(path)
    array = (
        df.values
    )  # array(["'Acta2'", 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=object)

    codebook = [np.array(x[1:], dtype=np.float32) for x in array]
    rna_name = [x[0].replace("'", "") for x in array]

    if bg:
        codebook = codebook + [np.zeros(15, dtype=np.float32)]
        rna_name =  rna_name + ["background"]
        print("Add background to codebook")

    return np.stack(codebook, axis=0), np.array(rna_name)


def obtain_init_color(input_xys, hw, image):
    # input_xys: [N, 2]
    # hw: [H, W]
    # image: [H, W, c]

    input_coords = input_xys / torch.tensor(hw, dtype=input_xys.dtype, device=input_xys.device).reshape(1, 2)
    input_coords = (input_coords - 0.5) * 2
    input_coords = input_coords[None, None, ...] # [1, 1, N, 2]

    image = image.permute(2, 0, 1)[None, ...] # (1, 15, H, W)

    # (1, 15, H, W), (1, 1, N, 2) -> (1, 15, 1, N)
    color = grid_sample(image, input_coords, align_corners=True) # (1, 15, 1, N)

    return color[0, :, 0, :].permute(1, 0) # (N, 15)


def filter_by_background(xys, colors, hw, image, th=0.05):
    gt_color = obtain_init_color(xys, hw, image) # (N, 15)
    weight_color = gt_color.max(dim=-1, keepdim=True)[0] # (N, 1)

    weight_color = (weight_color - th > 0).to(gt_color.dtype) # (N, 1)
    colors = colors * weight_color

    return colors


def write_to_csv(xys, scores, hw, rna_index, rna_name, path, score_th=0): #default 0
    mask = (xys[:, 0] >= 0) & (xys[:, 0] < hw[0]) & (xys[:, 1] >= 0) & (xys[:, 1] < hw[1])
    mask = mask & (scores[:, 0] > score_th)

    xys = xys[mask]

    xys = torch.round(xys).to(torch.int16)
    px = xys[:, 0].cpu().numpy()
    py = xys[:, 1].cpu().numpy()

    scores = scores[mask].cpu().numpy() # (n, k)
    rna_index = rna_index[mask].cpu().numpy()
    rna_name = rna_name[mask.cpu().numpy()]

    df = pd.DataFrame({
        "x": px, "y": py, "index": rna_index, "rna": rna_name, 
        "score": scores[:, 0], "cos_score": scores[:, 1], "max_color": scores[:, 2],
    })

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if score_th == 1:
        df.to_csv(path, index=False)
    else:
        df.to_csv(path.replace('.csv', f'_{score_th:.2f}.csv'), index=False)



if __name__ == "__main__":
    # load_ckpt_write_to_csv(ckpt_path='outputs/ablation_baseline',img_path=Path("data/1213_demo_data_v2/raw1"),codebook_path=Path("data/codebook.xlsx"))
    pass
