# script example to get cos_distance_error_map
# python3 vis_errormap.py --csv_path outputs/errormap/output_all.csv \
#     --img_path data/IM41340regi_192 \
#     --model_path outputs/errormap/params_20000.pth \

import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image

from utils import draw_results, count_vaild_pixel, count_vaild_class
from preprocess import images_to_tensor
from visualize import view_points
from gsplat2d import *

def normalize_image(image):
    # Scale values to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    return image

def tensor_to_image(tensor):
    # Clamp values to [0, 1] and convert to uint8
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()
    tensor = tensor.permute(2, 0, 1)
    # Convert tensor to PIL image
    image = TF.to_pil_image(tensor)
    return image

def read_and_vis_results(csv_path, code_book_path, img_path, model_path, pos_threshold=20):
    df = pd.read_csv(csv_path)

    px = df["x"].values
    py = df["y"].values
    pred_name = df["Class name"].values
    scores = df["cos_simi"].values
    index = df["Class index"].values

    alphas = df["15D value"].values
    alphas = [list(map(lambda y: float(y), x.split(' '))) for x in alphas]
    alphas = np.array(list(alphas))

    images = images_to_tensor(img_path).numpy()

    #(gzx):改用最大值而不是mean
    gt_image = images.max(axis=2)
    ref_scores = np.clip(gt_image[(py,px)] * 255.0 - pos_threshold, 0.0 , 10.0) / 10.0
    scores = scores * ref_scores
    image = np.tile(gt_image[..., None], [1, 1, 3])

    plt.subplot(1, 2, 2)
    seleted_idx = scores != 0
    ori_scores = scores
    scores = scores[seleted_idx]

    if model_path != None:
        device = torch.device("cuda:0")

        model = torch.load(model_path, map_location=torch.device(device))

        xys = model["xys"]
        rgbs = model["rgbs"]
        persistent_mask = model["persistent_mask"]
        opacities = model["opacities"]
        depth = model["depth"]
        sigma_x = model["sigma_x"]
        sigma_y = model["sigma_y"]
        rho = model["rho"]
        H, W = model["H"], model["W"]
        xys = torch.clamp(torch.tanh(xys[persistent_mask]) * 1.02, -1, 1)
        rho = torch.sigmoid(rho[persistent_mask])
        sigma_x = torch.sigmoid(sigma_x[persistent_mask]) * 5
        sigma_y = torch.sigmoid(sigma_y[persistent_mask]) * 5
        persist_rgbs = torch.sigmoid(rgbs[persistent_mask])
        persist_opacities = opacities[persistent_mask]

        depths, radii, conics, num_tiles_hit = project_gaussians_2D(
                sigma_x, sigma_y, rho, xys.shape[0], depth, device=device
            )
        torch.cuda.synchronize()
        scoresgpu = torch.tensor(ori_scores, dtype=torch.float32).to(device)
        scoresgpu = 1 - scoresgpu
        cmap = plt.cm.jet
        scorescpu = scoresgpu.clone().cpu()
        rgb_color = cmap(scorescpu, bytes=True)[:, :3]
        rgb_color_after = cmap(scorescpu[seleted_idx], bytes=True)[:, :3]
        ab_xys = (xys + 1) / 2 * H
        colors = torch.tensor(rgb_color).to(device)
        colors_after = torch.tensor(rgb_color_after).to(device)
        background = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        out_img_before = rasterize_gaussians(
            ab_xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            colors,
            persist_opacities,
            H,
            W,
            16,
            background
        )
        out_img_after = rasterize_gaussians(
            ab_xys[seleted_idx],
            depths[seleted_idx],
            radii[seleted_idx],
            conics[seleted_idx],
            num_tiles_hit,
            colors_after,
            persist_opacities[seleted_idx],
            H,
            W,
            16,
            background
        )
        torch.cuda.synchronize()

        out_img_before_normalized = normalize_image(out_img_before)
        output_image1 = tensor_to_image(out_img_before_normalized)
        output_image1.save("before_image.png")

        out_img_before_normalized = normalize_image(out_img_after)
        output_image2 = tensor_to_image(out_img_before_normalized)
        output_image2.save("after_image.png")

        gt_image_pil = Image.fromarray((image * 255).astype(np.uint8))
        result_image1 = Image.blend(gt_image_pil, output_image1, alpha=0.4)
        result_image2 = Image.blend(gt_image_pil, output_image2, alpha=0.4)
        result_image1.save("before_image_blend.png")
        result_image2.save("after_image_blend.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=Path("outputs/output_2304.csv"))
    parser.add_argument("--book_path", type=str, default=Path("data/codebook.xlsx"))
    parser.add_argument("--img_path", type=str, default=Path("../data/IM41340_processed"))
    parser.add_argument("--model_path", type=str, default=Path("outputs/errormap/params_20000.pth"))
    parser.add_argument("--pos_threshold", type=float, default=20)
    arg=parser.parse_args()

    read_and_vis_results(
        csv_path=arg.csv_path, 
        code_book_path = arg.book_path,
        img_path=arg.img_path,
        model_path = arg.model_path, 
        pos_threshold=arg.pos_threshold
    )
    