import os
import torch

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from skimage import filters


# (zwx)
def preprocess_data(dir_path: Path):
    image_paths = [dir_path / f"{i}.png" for i in range(1, 16)]
    ths = 13  # 更改二值化阈值
    pp = []
    for full_path in image_paths:
        img = cv2.imread(str(full_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), dtype=np.uint8)
        img_normalized = cv2.convertScaleAbs(img)
        usm = cv2.erode(img_normalized, kernel, iterations=1)
        # (gzx):用腐蚀之后的图像
        thresh = cv2.threshold(usm, ths, 255, cv2.THRESH_BINARY)[1]

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh.copy(), connectivity=8, ltype=cv2.CV_32S
        )
        # 加入连通块外接矩阵几何中心
        for i in range(1, num_labels):  # 从1开始，跳过背景区域
            x, y, width, height, area = stats[i]
            centre_x = np.int64(x + width // 2)
            centre_y = np.int64(y + height // 2)
            pp.append([centre_x, centre_y])
        for pi in centroids[1:]:
            px, py = np.int64(pi[0]), np.int64(pi[1])
            pp.append([px, py])
    pp = np.array(pp)
    # 去重
    unique_pp = np.unique(pp, axis=0)
    return len(unique_pp), unique_pp


def give_required_data(input_coords, image_size, image_array, device):
    # normalising pixel coordinates [-1,1]
    coords = torch.tensor(
        input_coords / [image_size[0], image_size[1]], device=device
    ).float()  # , dtype=torch.float32
    center_coords_normalized = torch.tensor(
        [0.5, 0.5], device=device
    ).float()  # , dtype=torch.float32
    coords = (center_coords_normalized - coords) * 2.0
    # Fetching the colour of the pixels in each coordinates
    colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_on_cpu = [
        tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in colour_values
    ]
    colour_values_np = np.array(colour_values_on_cpu)
    colour_values_tensor = torch.tensor(
        colour_values_np, device=device
    ).float()  # , dtype=torch.float32

    return colour_values_tensor, coords


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
def write_to_csv(pixel_coords, alpha, save_path, h, w, image,codebook_path = 'data/codebook.xlsx'):
    codebook = read_codebook(path=codebook_path)
    codebook = torch.tensor(codebook, device=alpha.device, dtype=alpha.dtype)
    rna_name = read_codebook_name(path=codebook_path)

    pred_code = alpha

    scores, indexs = get_index_cos(pred_code, codebook)  # (num_samples,)
    indexs = indexs.data.cpu().numpy()
    scores = scores.data.cpu().numpy()
    pred_name = rna_name[indexs]  # (num_samples,)

    px = pixel_coords.data.cpu().numpy()[:, 0]
    py = pixel_coords.data.cpu().numpy()[:, 1]

    px = px.astype(np.int16)
    py = py.astype(np.int16)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)

    df = pd.DataFrame(
        {"x": px, "y": py, "gene": pred_name, "index": indexs, "score": scores}
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.7))
    print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.85))
    print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.98))

    print(
        "number of vaild class (th=0.8)",
        count_vaild_class(score=scores, class_index=indexs, th=0.7),
    )
    print(
        "number of vaild class (th=0.9)",
        count_vaild_class(score=scores, class_index=indexs, th=0.85),
    )
    print(
        "number of vaild class (th=0.95)",
        count_vaild_class(score=scores, class_index=indexs, th=0.98),
    )

    image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

    draw_results(
        image, px, py, pred_name, scores, save_path.replace(".csv", "_vis_blank.png")
    )


# (zwx)
def write_to_csv_hamming(
    pixel_coords,
    alpha,
    save_path,
    h,
    w,
    image,
    codebook_path="../data/codebook.xlsx",
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
            "gene": pred_name,
            "index": (min_index + 1).flatten().data.cpu().numpy(),
            "min_distance_value": min_distance_values,
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

    image = np.tile(image.cpu().numpy()[..., None], [1, 1, 3])

    draw_results(
        image,
        px,
        py,
        pred_name,
        1 - min_distance_values,
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
    ax.scatter(px, py, alpha=scores, cmap=cm.jet, s=2)

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


def read_and_vis_results(csv_path):
    df = pd.read_csv(csv_path)

    px = df["x"].values
    py = df["y"].values
    pred_name = df["gene"].values
    scores = df["score"].values

    from PIL import Image

    images = []
    for i in range(1, 16):
        image = Image.open(f"data/1213_demo_data_v2/raw1/{i}.png").convert("RGB")
        image = np.array(image) / 255.0
        images.append(image)

    image = np.stack(images, axis=2)  # (h, w, 15)
    image = image.mean(axis=2)  # (h, w, 1)
    image = (image - image.min()) / (image.max() - image.min())

    draw_results(
        image,
        px,
        py,
        pred_name,
        scores,
        csv_path.replace(".csv", "_vis_read_all_image.png"),
    )

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


if __name__ == "__main__":
    read_and_vis_results(csv_path="outputs/codecos_0.001_vis/output.csv")
