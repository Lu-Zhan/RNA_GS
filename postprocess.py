import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from utils import draw_results, count_vaild_pixel, count_vaild_class
from preprocess import images_to_tensor
from visualize import view_points


def read_and_vis_results(csv_path, img_path, pos_threshold=20):
    df = pd.read_csv(csv_path)

    px = df["x"].values
    py = df["y"].values
    pred_name = df["Class name"].values
    scores = df["cos_simi"].values
    index = df["Class index"].values

    alphas = df["15D value"].values
    alphas = [list(map(lambda y: float(y), x.split(' '))) for x in alphas]
    alphas = np.array(list(alphas))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=100)
    plt.title("scores")
    plt.ylim(0, 500)

    images = images_to_tensor(img_path).numpy()

    #(gzx):改用最大值而不是mean
    gt_image = images.max(axis=2)
    ref_scores = np.clip(gt_image[(py,px)] * 255.0 - pos_threshold, 0.0 , 10.0) / 10.0
    scores = scores * ref_scores
    image = np.tile(gt_image[..., None], [1, 1, 3])    

    plt.subplot(1, 2, 2)
    seleted_idx = scores != 0
    scores = scores[seleted_idx]

    plt.hist(scores, bins=100)
    plt.title("scores after postprocessing")
    plt.ylim(0, 500)
    
    plt.savefig(str(csv_path).replace(".csv", f"_hist_post{pos_threshold}.png"))

    # mdp visualization
    draw_results(
        image,
        px[seleted_idx],
        py[seleted_idx],
        pred_name[seleted_idx],
        scores,
        str(csv_path).replace(".csv", f"_post{pos_threshold}.png"),
    )

    # view points
    view_points(
        torch.tensor(images),
        px[seleted_idx],
        py[seleted_idx],
        alphas[seleted_idx],
        str(csv_path).replace(".csv", f"_post{pos_threshold}_points.png"),
    )
    
    df = pd.DataFrame(
        {
            "x": px[seleted_idx],
            "y": py[seleted_idx],
            "Class name": pred_name[seleted_idx],
            "Class index": index[seleted_idx],
            "cos_simi": scores,
            "Sigma_x": df["Sigma_x"][seleted_idx],
            "Sigma_y": df["Sigma_y"][seleted_idx],
            "15D value": df["15D value"][seleted_idx],
        }
    )

    df.to_csv(str(csv_path).replace(".csv", f"_post{pos_threshold}.csv"), index=False)

    print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.8))
    print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.9))
    print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.95))

    print(
        "number of vaild class (th=0.0001)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.0001),
    )
    print(
        "number of vaild class (th=0.5)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.5),
    )
    print(
        "number of vaild class (th=0.8)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.8),
    )


def read_and_vis_results_v2(csv_path, img_path, pos_threshold=20):
    df = pd.read_csv(csv_path)

    px = df["x"].values
    py = df["y"].values
    pred_name = df["class"].values
    scores = df["score"].values
    index = df["index"].values

    # alphas = df["15D value"].values
    # alphas = [list(map(lambda y: float(y), x.split(' '))) for x in alphas]
    # alphas = np.array(list(alphas))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=100)
    plt.title("scores")
    plt.ylim(0, 500)

    images = images_to_tensor(img_path).numpy()

    # #(gzx):改用最大值而不是mean
    gt_image = images.max(axis=2)
    # ref_scores = np.clip(gt_image[(py,px)] * 255.0 - pos_threshold, 0.0 , 10.0) / 10.0
    # scores = scores * ref_scores
    image = np.tile(gt_image[..., None], [1, 1, 3])    

    plt.subplot(1, 2, 2)
    # seleted_idx = scores != 0
    # scores = scores[seleted_idx]

    plt.hist(scores, bins=100)
    plt.title("scores after postprocessing")
    plt.ylim(0, 500)
    
    plt.savefig(str(csv_path).replace(".csv", f"_hist_post{pos_threshold}.png"))

    # mdp visualization
    draw_results(
        image,
        px,
        py,
        pred_name,
        scores,
        str(csv_path).replace(".csv", f"_post{pos_threshold}.png"),
    )

    # # view points
    # view_points(
    #     torch.tensor(images),
    #     px[seleted_idx],
    #     py[seleted_idx],
    #     alphas[seleted_idx],
    #     str(csv_path).replace(".csv", f"_post{pos_threshold}_points.png"),
    # )
    
    df = pd.DataFrame(
        {
            "x": px,
            "y": py,
            "class": pred_name,
            "index": index,
            "score": scores,
        }
    )

    df.to_csv(str(csv_path).replace(".csv", f"_post{pos_threshold}.csv"), index=False)

    print("number of vaild point (th=0.8)", count_vaild_pixel(score=scores, th=0.8))
    print("number of vaild point (th=0.9)", count_vaild_pixel(score=scores, th=0.9))
    print("number of vaild point (th=0.95)", count_vaild_pixel(score=scores, th=0.95))

    print(
        "number of vaild class (th=0.0001)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.0001),
    )
    print(
        "number of vaild class (th=0.5)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.5),
    )
    print(
        "number of vaild class (th=0.8)",
        count_vaild_class(score=scores, class_index=pred_name, th=0.8),
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=Path("outputs/output_2304.csv"))
    parser.add_argument("--img_path", type=str, default=Path("../data/IM41340_processed"))
    parser.add_argument("--pos_threshold", type=float, default=20)
    arg=parser.parse_args()

    read_and_vis_results(
        csv_path=arg.csv_path, 
        img_path=arg.img_path,
        pos_threshold=arg.pos_threshold
    )
    