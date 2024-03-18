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
    rna_name = [x[0] for x in array]

    if bg:
        codebook = [np.zeros(15, dtype=np.float32)] + codebook
        rna_name = ["background"] + rna_name
        print("Add background to codebook")

    return np.stack(codebook, axis=0), np.array(rna_name)


def obtain_init_color(input_xys, hw, image):
    # input_xys: [N, 2]
    # hw: [H, W]
    # image: [H, W, 15]

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


def write_to_csv(xys, scores, hw, rna_index, rna_name, path):
    mask = (xys[:, 0] >= 0) & (xys[:, 0] < hw[0]) & (xys[:, 1] >= 0) & (xys[:, 1] < hw[1])
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
    df.to_csv(path, index=False)


#     codebook = read_codebook(path=codebook_path)
#     codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
#     rna_name = read_codebook_name(path=codebook_path)

#     pred_code = alpha

#     scores, indexs = get_index_cos(pred_code, codebook)  # (num_samples,)

#     px = pixel_coords.data.cpu().numpy()[:, 0]
#     py = pixel_coords.data.cpu().numpy()[:, 1]

#     px = px.astype(np.int16)
#     py = py.astype(np.int16)
#     px = np.clip(px, 0, w - 1)
#     py = np.clip(py, 0, h - 1)
    
#     #(gzx):后处理
#     print(pos_threshold)
#     if post_processing == False:
#         indexs = indexs.data.cpu().numpy()
#         scores = scores.data.cpu().numpy()
#         origin_scores = scores.copy()
#     else:
#         ref_scores = torch.clamp(ref.max(dim=2)[0][(py, px)] * 255.0 - pos_threshold, 0.0, 10.0) / 10.0
#         # for i in range(20):
#         #     print(ref_scores[i*20:i*20+20])

#         scores, indexs = get_index_cos(pred_code, codebook)  # (num_samples,)
#         scores = scores.data.cpu().numpy()
#         ref_scores = ref_scores.data.cpu().numpy()
#         origin_scores = scores.copy()
#         indexs = indexs.data.cpu().numpy()
        
#         total_scores = ref_scores * scores
#         scores = total_scores
#         # scores = total_scores.data.cpu().numpy() 

        
#     pred_name = rna_name[indexs]  # (num_samples,)
     
#     df = pd.DataFrame(
#         {"x": px, "y": py, "gene": pred_name, "index": indexs, "score": scores}
#     )

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     df.to_csv(save_path, index=False)

#     print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.8))
#     print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.9))
#     print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.95))

#     print(
#         "number of vaild class (th=0.8)",
#         count_vaild_class(score=scores, class_index=indexs, th=0.8),
#     )
#     print(
#         "number of vaild class (th=0.9)",
#         count_vaild_class(score=scores, class_index=indexs, th=0.9),
#     )
#     print(
#         "number of vaild class (th=0.95)",
#         count_vaild_class(score=scores, class_index=indexs, th=0.95),
#     )

#     image = ref.max(dim=2)[0] 
#     image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

#     draw_results(
#         image, px, py, pred_name, origin_scores, save_path.replace(".csv", "_no_postprocess.png")
#     )

#     if post_processing:
#         draw_results(
#             image,
#             px,
#             py,
#             pred_name,
#             scores,
#             save_path.replace(".csv", f"_post{pos_threshold}.png"),
#         )


# def read_codebook_name(path):
#     data_frame = pd.read_excel(path)
#     column_data = data_frame.iloc[0:, 0].astype(str).values
#     return column_data


# def get_index_cos(pred_code, codebook):
#     simi = pred_code @ codebook.T  # (num_samples, 15) @ (15, 181) = (num_samples, 181)

#     # (n, 181) / (n, 1) / (1, 181)
#     simi = (
#         simi
#         / torch.norm(pred_code, dim=-1, keepdim=True)
#         / torch.norm(codebook, dim=-1, keepdim=True).T
#     )

#     max_value, index = torch.max(simi, dim=-1)  # (num_samples, )

#     max_value = (max_value + 1) / 2

#     return max_value, index




# # cyy: 增加从ckpt读取并输出csv
# # usage: load_ckpt_write_to_csv(ckpt_path='outputs/ablation_baseline',img_path=Path("data/1213_demo_data_v2/raw1"),codebook_path=Path("data/codebook.xlsx"))
# def load_ckpt_write_to_csv(ckpt_path,img_path,codebook_path):
#     params_file = os.path.join(ckpt_path, "params.pth")
#     loaded_params = torch.load(params_file)
#     means = loaded_params["means"]
#     scales = loaded_params["scales"]
#     quats = loaded_params["quats"]
#     rgbs = loaded_params["rgbs"]
#     opacities = loaded_params["opacities"]
#     viewmat = loaded_params["viewmat"]
#     persistent_mask = loaded_params["persistent_mask"]
#     focal = loaded_params["focal"]
#     H = loaded_params["H"]
#     W = loaded_params["W"]
#     tile_bounds = loaded_params["tile_bounds"]
#     persist_means  = means [persistent_mask]
#     persist_scales = scales[persistent_mask]
#     persist_quats  = quats [persistent_mask]
#     persist_rgbs   = rgbs  [persistent_mask]
#     xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
#                 persist_means,
#                 persist_scales,
#                 1,
#                 persist_quats,
#                 viewmat,
#                 viewmat,
#                 focal,
#                 focal,
#                 W / 2,
#                 H / 2,
#                 H,
#                 W,
#                 tile_bounds,
#             )
#     alpha = torch.sigmoid(persist_rgbs)
#     gt_image = images_to_tensor(img_path)
#     write_to_csv_all(
#             pixel_coords=xys,
#             sigma=conics,
#             alpha=alpha,
#             save_path=f"{ckpt_path}/output_all_ckpt.csv",
#             h=H,
#             w=W,
#             ref=gt_image,
#             codebook_path = codebook_path,
#         )


# # cyy: 增加输出ppt格式的csv
# def write_to_csv_all(
#     pixel_coords,
#     alpha,
#     sigma,
#     save_path,
#     h,
#     w,
#     ref=None,
#     pos_threshold = 20.0,
#     codebook_path="./data/codebook.xlsx",
# ):
#     codebook = read_codebook(path=codebook_path)
#     codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
#     expanded_codebook = codebook.unsqueeze(0)
#     rna_name = read_codebook_name(path=codebook_path)
#     sigma_x, sigma_y = obtain_sigma_xy(sigma)
#     # losses=['cos','mean','median','li']
#     losses=['cos']
#     for loss in losses:
#         if loss=="cos":
#             pred_code = alpha
#             min_distance_values, min_index = get_index_cos(pred_code, codebook)  # (num_samples,)
#         if loss == "mean":
#             sorted_ranks, _ = torch.sort(alpha, dim=0)
#             # 计算每列前70%的阈值
#             threshold = sorted_ranks[int(0.7 * alpha.size(0))]
#             pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
#         if loss == "median":
#             threshold = torch.median(alpha)
#             pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
#         if loss == "li":
#             pred_code_bin = torch.zeros_like(alpha)
#             for i in range(alpha.shape[1]):
#                 # 取出同一张图片的alpha
#                 image = alpha[:, i]
#                 image_np = np.asarray(image.detach().cpu())
#                 best_threshold = filters.threshold_li(image_np)
#                 # 与阈值进行比较
#                 binary_image = image > best_threshold
#                 pred_code_bin[:, i] = binary_image
#         if loss!="cos":
#             expanded_pred_code = pred_code_bin.unsqueeze(1)
#             hamming_distance = (expanded_pred_code != expanded_codebook).sum(
#                 dim=2
#             ).float() / codebook.shape[1]
#             # min_index = torch.argmin(hamming_distance, dim=1, keepdim=True)
#             min_distance_values, min_index = torch.min(hamming_distance, dim=1, keepdim=True)
#         px = pixel_coords.data.cpu().numpy()[:, 0]
#         py = pixel_coords.data.cpu().numpy()[:, 1]
#         px = px.astype(np.int16)
#         py = py.astype(np.int16)
#         px = np.clip(px, 0, w - 1)
#         py = np.clip(py, 0, h - 1)
#         if loss=="cos":
#             pred_name = [
#                 rna_name[int(min_index[x].item())].replace("'", "")
#                 for x in range(min_index.shape[0])
#             ]
#             df = pd.DataFrame(
#                 {
#                     "x": px,
#                     "y": py,
#                     "Class index": (min_index + 1).flatten().data.cpu().numpy(),
#                     "Class name": np.array(pred_name),
#                     # "cos_score": (min_distance_values).detach().cpu().numpy(),
#                 }
#             )
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             ref_scores = torch.clamp(ref.max(dim=2)[0][(py,px)]*255.0 - pos_threshold, 0.0 ,10.0)/10.0
#             min_distance_values = (ref_scores.cpu()) * (min_distance_values.cpu())
#             df['cos_simi']=min_distance_values.detach().cpu().numpy()
#             df['Sigma_x']=sigma_x.detach().cpu().numpy()
#             df['Sigma_y']=sigma_y.detach().cpu().numpy()
#             compressed_alpha = [' '.join([str(element) for element in row.tolist()]) for row in alpha]
#             df['15D value']=compressed_alpha
#         else:
#             min_distance_values=min_distance_values.flatten()
#             df[f'hm_{loss}_score']=(1-min_distance_values).cpu().numpy()
#             hm_score = 1 - min_distance_values
#             ref_scores = torch.clamp(ref.max(dim=2)[0][(py,px)]*255.0 - pos_threshold, 0.0 ,10.0)/10.0
#             hm_score = ref_scores * hm_score
#             df[f'hm_{loss}_score_post']=(hm_score).cpu().numpy()
        
#     df.to_csv(save_path, index=True)


# # (zwx)
# def write_to_csv_hamming(
#     pixel_coords,
#     alpha,
#     save_path,
#     h,
#     w,
#     image,
#     ref=None,
#     post_processing=False,
#     pos_threshold = 0.0,
#     codebook_path="./data/codebook.xlsx",
#     loss="mean",
# ):
#     codebook = read_codebook(path=codebook_path)
#     codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
#     expanded_codebook = codebook.unsqueeze(0)
#     rna_name = read_codebook_name(path=codebook_path)

#     if loss == "mean":
#         sorted_ranks, _ = torch.sort(alpha, dim=0)
#         # 计算每列前70%的阈值
#         threshold = sorted_ranks[int(0.7 * alpha.size(0))]
#         pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
#     elif loss == "median":
#         threshold = torch.median(alpha)
#         pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
#     elif loss == "li":
#         pred_code_bin = torch.zeros_like(alpha)
#         for i in range(alpha.shape[1]):
#             # 取出同一张图片的alpha
#             image = alpha[:, i]
#             image_np = np.asarray(image.detach().cpu())
#             best_threshold = filters.threshold_li(image_np)
#             # 与阈值进行比较
#             binary_image = image > best_threshold
#             pred_code_bin[:, i] = binary_image
#     expanded_pred_code = pred_code_bin.unsqueeze(1)
#     hamming_distance = (expanded_pred_code != expanded_codebook).sum(
#         dim=2
#     ).float() / codebook.shape[1]
#     # min_index = torch.argmin(hamming_distance, dim=1, keepdim=True)
#     min_distance_values, min_index = torch.min(hamming_distance, dim=1, keepdim=True)
#     pred_name = [
#         rna_name[int(min_index[x].item())].replace("'", "")
#         for x in range(min_index.shape[0])
#     ]

#     px = pixel_coords.data.cpu().numpy()[:, 0]
#     py = pixel_coords.data.cpu().numpy()[:, 1]

#     px = px.astype(np.int16)
#     py = py.astype(np.int16)
#     px = np.clip(px, 0, w - 1)
#     py = np.clip(py, 0, h - 1)

#     df = pd.DataFrame(
#         {
#             "x": px,
#             "y": py,
#             "gene": np.array(pred_name),
#             "index": (min_index + 1).flatten().data.cpu().numpy(),
#             # "min_distance_value": min_distance_values[:, 0].cpu().numpy(),
#             "score": (1-min_distance_values[:, 0]).cpu().numpy(),
#         }
#     )

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     df.to_csv(save_path, index=False)
    
#     if post_processing:
#         ref_scores = torch.clamp(ref.max(dim=2)[0][(py,px)]*255.0 - pos_threshold, 0.0 ,10.0)/10.0
#         # for i in range(20):
#         #     print(ref_scores[i*20:i*20+20])
#         min_distance_values = 1 - min_distance_values
        
#         min_distance_values = 1 - ref_scores * min_distance_values

#     image = ref.max(dim=2)[0]
#     image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

#     draw_results(
#         image,
#         px,
#         py,
#         pred_name,
#         1 - min_distance_values.cpu().numpy(),
#         save_path.replace(".csv", "_vis_blank.png"),
#     )


# def draw_results(image, px, py, pred_name, scores, save_path):
#     # draw scatter plot on the image no 1, score is the alpha value

#     import matplotlib.pyplot as plt
#     from matplotlib import cm

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image)
#     scores = np.clip(scores, 0, 1)
#     scores = scores
#     # # scores = (scores - 0.95) / 0.05
#     # # scores = np.clip(scores, 0, 1)
#     # ax.scatter(py, px, alpha=scores, cmap=cm.jet, s=2)
#     ax.scatter(px, py,color="red", alpha=scores, cmap=cm.jet, s=2)

#     # for i, txt in enumerate(pred_name):
#     #     ax.annotate(txt, (px[i], py[i]))

#     plt.savefig(save_path)


# def count_vaild_pixel(score, th=0.9):
#     return (score > th).sum()


# def count_vaild_class(score, class_index, th=0.9):
#     vaild_index = score > th

#     class_index = class_index[vaild_index]

#     # count number of different classes
#     unique, counts = np.unique(class_index, return_counts=True)

#     return len(counts)


# def read_and_vis_results(csv_path,img_path,pos_threshold=20):
#     df = pd.read_csv(csv_path)

#     px = df["x"].values
#     py = df["y"].values
#     pred_name = df["gene"].values
#     scores = df["score"].values
#     index = df["index"].values

#     from PIL import Image
    
    
#     #（gzx）：画分布图
#     #plot 1:

#     plt.subplot(1, 2, 1)
#     plt.hist(scores, bins=100)
#     plt.title("scores")
#     plt.ylim(0, 500)

#     images = []
#     for i in range(1, 16):
#         image = Image.open(f"{img_path}/{i}.png").convert("L")
#         image = np.array(image) / 255.0
#         images.append(image)

#     image = np.stack(images, axis=2)  # (h, w, 15)
#     # image = image.mean(axis=2)  # (h, w, 1)
#     # image = (image - image.min()) / (image.max() - image.min())

#     #(gzx):改用最大值而不是mean
#     gt_image = image.max(axis=2)
#     ref_scores = np.clip(gt_image[(py,px)]*255.0 - pos_threshold, 0.0 ,10.0) / 10.0
#     scores = scores * ref_scores
#     image = np.tile(gt_image[..., None], [1, 1, 3])
    
#     #plot 2:

#     plt.subplot(1, 2, 2)

#     scores = scores[scores != 0]

#     plt.hist(scores, bins=100)
#     plt.title("scores after postprocessing")
#     plt.ylim(0, 500)
    
#     plt.savefig(str(csv_path).replace(".csv", f"_hist_post{pos_threshold}.png"))

#     draw_results(
#         image,
#         px,
#         py,
#         pred_name,
#         scores,
#         str(csv_path).replace(".csv", f"_post{pos_threshold}.png"),
#     )
    
#     df = pd.DataFrame(
#         {
#             "x": px,
#             "y": py,
#             "gene": pred_name,
#             "index": index,
#             "score": scores,
#         }
#     )

#     df.to_csv(str(csv_path).replace(".csv", f"_post{pos_threshold}.csv"), index=False)

#     print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.7))
#     print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.85))
#     print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.99))

#     print(
#         "number of vaild class (th=0.0001)",
#         count_vaild_class(score=scores, class_index=pred_name, th=0.0001),
#     )
#     print(
#         "number of vaild class (th=0.5)",
#         count_vaild_class(score=scores, class_index=pred_name, th=0.5),
#     )
#     print(
#         "number of vaild class (th=0.8)",
#         count_vaild_class(score=scores, class_index=pred_name, th=0.8),
#     )



if __name__ == "__main__":
    # load_ckpt_write_to_csv(ckpt_path='outputs/ablation_baseline',img_path=Path("data/1213_demo_data_v2/raw1"),codebook_path=Path("data/codebook.xlsx"))
    pass
