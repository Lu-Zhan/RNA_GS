import numpy as np
import matplotlib.pyplot as plt

from torch.nn.functional import interpolate

def view_output(pred, gt, resize=(192, 192)):
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