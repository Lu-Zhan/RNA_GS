import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.nn.functional import interpolate


def view_positions(points_xy, bg_image, alpha=1, s=1, prefix=""):
    if len(points_xy) == 0:
        fig, ax = plt.subplots()
        ax.imshow(bg_image, cmap="gray")
        ax.axis("off")

        plt.title(f"{prefix}No points")
        plt.tight_layout()
        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close()

        return Image.fromarray(data)
    # points_xy: [n, 2], bg_image: [h, w, 1], alpha: [n, 15]
    
    # select points within the image
    mask = (points_xy[:, 0] >= 0) & (points_xy[:, 0] < bg_image.shape[1]) & (points_xy[:, 1] >= 0) & (points_xy[:, 1] < bg_image.shape[0])
    points_xy = points_xy[mask]

    alpha = alpha[mask]
    # alpha = np.max(alpha, axis=-1)
    # alpha = alpha / (alpha.max() + 1e-8)

    fig, ax = plt.subplots()
    ax.imshow(bg_image, cmap="gray")
    ax.scatter(points_xy[:, 0], points_xy[:, 1], c="r", s=s, alpha=alpha)
    ax.axis("off")
    plt.title(f"{prefix}Points: {len(points_xy)}, top0.9: {len(alpha[alpha > 0.8])}, top0.7: {len(alpha[alpha > 0.5])}, top0.5: {len(alpha[alpha > 0.2])}")

    plt.tight_layout()
    # plt.subplots_adjust(
    #     left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.15
    # )
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()

    return Image.fromarray(data)


def view_recon(pred, gt, resize=(192, 192)):
    # view pred and gt images in 15 groups.
    # for each group, pred on left, gt on right, pred: [h, w, 15], gt: [h, w, 15], using heatmap
    # resize to 192 x 192
    pred = interpolate(pred.permute(2, 0, 1)[None, ...], size=resize, mode="bilinear", align_corners=False)
    gt = interpolate(gt.permute(2, 0, 1)[None, ...], size=resize, mode="bilinear", align_corners=False)

    pred = pred[0].permute(1, 2, 0)
    gt = gt[0].permute(1, 2, 0)
    
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    fig, axs = plt.subplots(6, 5, figsize=(15, 15))
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    # plt.margins(0, 0)

    # vmax = max(pred.max(), gt.max())

    for i in range(3):
        for j in range(5):
            # pred
            axs[i * 2 + 1, j].imshow(pred[..., i * 5 + j], cmap="jet", interpolation="nearest") #, vmin=0, vmax=vmax) 
            axs[i * 2 + 1, j].axis("off")
            axs[i * 2 + 1, j].set_title(f"Pred {i * 5 + j}")

            # gt
            axs[i * 2, j].imshow(gt[..., i * 5 + j], cmap="jet", interpolation="nearest") #, vmin=0, vmax=vmax) 
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