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



def read_codebook(path):
    df = pd.read_excel(path)
    array = (
        df.values
    )  # array(["'Acta2'", 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0], dtype=object)

    codebook = [np.array(x[1:], dtype=np.float32) for x in array]

    return np.stack(codebook, axis=0)  # (181, 15)


def read_codebook_name(path):
    data_frame = pd.read_excel(path)
    column_data = data_frame.iloc[0:, 0].astype(str).values
    return column_data


def get_index_cos(pred_code, codebook):
    simi = pred_code @ codebook.T  # (num_samples, 15) @ (15, 181) = (num_samples, 181)

    # (n, 181) / (n, 1) / (1, 181)
    simi = (
        simi
        / torch.norm(pred_code, dim=-1, keepdim=True)
        / torch.norm(codebook, dim=-1, keepdim=True).T
    )

    max_value, index = torch.max(simi, dim=-1)  # (num_samples, )

    max_value = (max_value + 1) / 2

    return max_value, index


#(gzx):读入的codebook_path
def write_to_csv(
        pixel_coords,
        alpha, 
        save_path,
        h, w,
        image,
        ref=None,
        post_processing=False,
        pos_threshold=0.0,
        codebook_path='data/codebook.xlsx'
    ):

    codebook = read_codebook(path=codebook_path)
    codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
    rna_name = read_codebook_name(path=codebook_path)

    pred_code = alpha

    scores, indexs = get_index_cos(pred_code, codebook)  # (num_samples,)

    px = pixel_coords.data.cpu().numpy()[:, 0]
    py = pixel_coords.data.cpu().numpy()[:, 1]

    px = px.astype(np.int16)
    py = py.astype(np.int16)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    
    #(gzx):后处理
    print(pos_threshold)
    if post_processing == False:
        indexs = indexs.data.cpu().numpy()
        scores = scores.data.cpu().numpy()
        origin_scores = scores.copy()
    else:
        ref_scores = torch.clamp(ref.max(dim=2)[0][(py, px)] * 255.0 - pos_threshold, 0.0, 10.0) / 10.0
        # for i in range(20):
        #     print(ref_scores[i*20:i*20+20])

        scores, indexs = get_index_cos(pred_code, codebook)  # (num_samples,)
        origin_scores = scores.copy()
        indexs = indexs.data.cpu().numpy()
        
        total_scores = ref_scores * scores
        scores = total_scores.data.cpu().numpy() 

        
    pred_name = rna_name[indexs]  # (num_samples,)
     
    df = pd.DataFrame(
        {"x": px, "y": py, "gene": pred_name, "index": indexs, "score": scores}
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.8))
    print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.9))
    print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.95))

    print(
        "number of vaild class (th=0.8)",
        count_vaild_class(score=scores, class_index=indexs, th=0.8),
    )
    print(
        "number of vaild class (th=0.9)",
        count_vaild_class(score=scores, class_index=indexs, th=0.9),
    )
    print(
        "number of vaild class (th=0.95)",
        count_vaild_class(score=scores, class_index=indexs, th=0.95),
    )

    image = ref.max(dim=2)[0] 
    image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

    draw_results(
        image, px, py, pred_name, origin_scores, save_path.replace(".csv", "_no_postprocess.png")
    )

    if post_processing:
        draw_results(
            image,
            px,
            py,
            pred_name,
            scores,
            save_path.replace(".csv", f"_post{pos_threshold}.png"),
        )


# (zwx)
def write_to_csv_hamming(
    pixel_coords,
    alpha,
    save_path,
    h,
    w,
    image,
    ref=None,
    post_processing=False,
    pos_threshold = 0.0,
    codebook_path="./data/codebook.xlsx",
    loss="mean",
):
    codebook = read_codebook(path=codebook_path)
    codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
    expanded_codebook = codebook.unsqueeze(0)
    rna_name = read_codebook_name(path=codebook_path)

    if loss == "mean":
        sorted_ranks, _ = torch.sort(alpha, dim=0)
        # 计算每列前70%的阈值
        threshold = sorted_ranks[int(0.7 * alpha.size(0))]
        pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
    elif loss == "median":
        threshold = torch.median(alpha)
        pred_code_bin = torch.where(alpha > threshold, torch.tensor(1), torch.tensor(0))
    elif loss == "li":
        pred_code_bin = torch.zeros_like(alpha)
        for i in range(alpha.shape[1]):
            # 取出同一张图片的alpha
            image = alpha[:, i]
            image_np = np.asarray(image.detach().cpu())
            best_threshold = filters.threshold_li(image_np)
            # 与阈值进行比较
            binary_image = image > best_threshold
            pred_code_bin[:, i] = binary_image
    expanded_pred_code = pred_code_bin.unsqueeze(1)
    hamming_distance = (expanded_pred_code != expanded_codebook).sum(
        dim=2
    ).float() / codebook.shape[1]
    # min_index = torch.argmin(hamming_distance, dim=1, keepdim=True)
    min_distance_values, min_index = torch.min(hamming_distance, dim=1, keepdim=True)
    pred_name = [
        rna_name[int(min_index[x].item())].replace("'", "")
        for x in range(min_index.shape[0])
    ]

    px = pixel_coords.data.cpu().numpy()[:, 0]
    py = pixel_coords.data.cpu().numpy()[:, 1]

    px = px.astype(np.int16)
    py = py.astype(np.int16)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    df = pd.DataFrame(
        {
            "x": px,
            "y": py,
            "gene": np.array(pred_name),
            "index": (min_index + 1).flatten().data.cpu().numpy(),
            # "min_distance_value": min_distance_values[:, 0].cpu().numpy(),
            "score": (1-min_distance_values[:, 0]).cpu().numpy(),
        }
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    # print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.7))
    # print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.85))
    # print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.98))

    # print(
    #     "number of vaild class (th=0.8)",
    #     count_vaild_class(score=scores, class_index=indexs, th=0.7),
    # )
    # print(
    #     "number of vaild class (th=0.9)",
    #     count_vaild_class(score=scores, class_index=indexs, th=0.85),
    # )
    # print(
    #     "number of vaild class (th=0.95)",
    #     count_vaild_class(score=scores, class_index=indexs, th=0.98),
    # )
    
    #(gzx):后处理
    
    if post_processing:
        ref_scores = torch.clamp(ref.max(dim=2)[0][(py,px)]*255.0 - pos_threshold, 0.0 ,10.0)/10.0
        # for i in range(20):
        #     print(ref_scores[i*20:i*20+20])
        min_distance_values = 1 - min_distance_values
        
        min_distance_values = 1 - ref_scores * min_distance_values

    image = ref.max(dim=2)[0]
    image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

    draw_results(
        image,
        px,
        py,
        pred_name,
        1 - min_distance_values.cpu().numpy(),
        save_path.replace(".csv", "_vis_blank.png"),
    )


def draw_results(image, px, py, pred_name, scores, save_path):
    # draw scatter plot on the image no 1, score is the alpha value

    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    scores = np.clip(scores, 0, 1)
    scores = scores
    # # scores = (scores - 0.95) / 0.05
    # # scores = np.clip(scores, 0, 1)
    # ax.scatter(py, px, alpha=scores, cmap=cm.jet, s=2)
    ax.scatter(px, py,color="red", alpha=scores, cmap=cm.jet, s=2)

    # for i, txt in enumerate(pred_name):
    #     ax.annotate(txt, (px[i], py[i]))

    plt.savefig(save_path)


def count_vaild_pixel(score, th=0.9):
    return (score > 0.9).sum()


def count_vaild_class(score, class_index, th=0.9):
    vaild_index = score > th

    class_index = class_index[vaild_index]

    # count number of different classes
    unique, counts = np.unique(class_index, return_counts=True)

    return len(counts)


def read_and_vis_results(csv_path,img_path,pos_threshold=20):
    df = pd.read_csv(csv_path)

    px = df["x"].values
    py = df["y"].values
    pred_name = df["gene"].values
    scores = df["score"].values
    index = df["index"].values

    from PIL import Image
    
    
    #（gzx）：画分布图
    #plot 1:

    plt.subplot(1, 2, 1)
    plt.hist(scores)
    plt.title("scores")


    images = []
    for i in range(1, 16):
        image = Image.open(f"{img_path}/{i}.png").convert("L")
        image = np.array(image) / 255.0
        images.append(image)

    image = np.stack(images, axis=2)  # (h, w, 15)
    # image = image.mean(axis=2)  # (h, w, 1)
    # image = (image - image.min()) / (image.max() - image.min())

    #(gzx):改用最大值而不是mean
    gt_image = image.max(axis=2)
    ref_scores = np.clip(gt_image[(py,px)]*255.0 - pos_threshold, 0.0 ,10.0)/10.0
    scores = scores * ref_scores
    image = np.tile(gt_image[..., None], [1, 1, 3])
    
    #plot 2:

    plt.subplot(1, 2, 2)
    plt.hist(scores)
    plt.title("scores after postprocessing")
    
    plt.savefig(str(csv_path).replace(".csv","_hist.png"))


    draw_results(
        image,
        px,
        py,
        pred_name,
        scores,
        str(csv_path).replace(".csv", f"_post{pos_threshold}.png"),
    )
    
    df = pd.DataFrame(
        {
            "x": px,
            "y": py,
            "gene": pred_name,
            "index": index,
            "score": scores,
        }
    )

    df.to_csv(str(csv_path).replace(".csv", f"_post{pos_threshold}.csv"), index=False)
    count_above_threshold = np.sum(scores > 0.5)
    print(f"大于0.5的数量: {count_above_threshold}")

    print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.7))
    print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.85))
    print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.99))

    print(
        "number of vaild class (th=0.8)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.7),
    )
    print(
        "number of vaild class (th=0.9)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.85),
    )
    print(
        "number of vaild class (th=0.99)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.999),
    )

# (zwx) PSNR after maximum density projection
def MDP_recon_psnr(img, gt_img):
    MDP_img = img.max(axis = 2).values
    MDP_gt_img = gt_img.max(axis = 2).values
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(MDP_img, MDP_gt_img)
    MDP_PSNR =  float(10 * torch.log10(1 / mse))
    return MDP_PSNR

if __name__ == "__main__":
    pass

