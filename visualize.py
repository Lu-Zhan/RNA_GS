import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 


def view_output(pred, gt):
    # view pred and gt images in 15 groups.
    # for each group, pred on left, gt on right, pred: [h, w, 15], gt: [h, w, 15], using heatmap
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    fig, axs = plt.subplots(6, 5, figsize=(15, 15))
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    # plt.margins(0, 0)

    for i in range(3):
        for j in range(5):
            # pred
            axs[i * 2 + 1, j].imshow(
                pred[..., i * 5 + j], cmap="jet", interpolation="nearest"
            )
            axs[i * 2 + 1, j].axis("off")
            axs[i * 2 + 1, j].set_title(f"Pred {i * 5 + j}")

            # gt
            axs[i * 2, j].imshow(
                gt[..., i * 5 + j], cmap="jet", interpolation="nearest"
            )
            axs[i * 2, j].axis("off")
            axs[i * 2, j].set_title(f"GT {i * 5 + j}")

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.15
    )
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()

    return data

def view_points(gt, px, py, scores, save_path):
    # view pred and gt images in 15 groups.
    # for each group, pred on left, gt on right, pred: [h, w, 15], gt: [h, w, 15], using heatmap
    gt = gt.detach().cpu().numpy()

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    # plt.margins(0, 0)
    scores = np.clip(scores, 0, 1)

    for i in range(4):
        for j in range(4):
            
            if i * 4 + j == 15:
                # sc
                axs[i , j].imshow(
                    gt[..., 0] * 0, cmap=plt.get_cmap('gray'), interpolation="nearest"
                )                
                axs[i , j].axis("off")
                axs[i , j].set_title(f"NOTHING")
                break

            # sc
            axs[i , j].imshow(
                gt[..., i * 4 + j], cmap=plt.get_cmap('gray'), interpolation="nearest"
            )
            
            axs[i , j].scatter(px, py, color="red", alpha=scores[:,i * 4 + j], cmap="jet", s=2)

            axs[i , j].axis("off")
            axs[i , j].set_title(f"SC {i * 4 + j}")

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.15
    )
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()

    Image.fromarray(data).save(save_path)