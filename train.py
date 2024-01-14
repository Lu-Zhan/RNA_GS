import math
import os
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
import wandb

from losses import (
    mse_loss, 
    l1_loss, 
    masked_mse_loss, 
    masked_l1_loss, 
    bg_loss, 
    ssim_loss, 
    codebook_cos_loss
)
from utils import write_to_csv, read_codebook


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        cfg: Optional[dict] = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

        self._init_gaussians()
        self.codebook = torch.tensor(read_codebook(cfg['codebook_path']), device=self.device)
        self.cfg = cfg  

        self.output_folder = os.path.join('outputs', self.cfg['exp_name'])
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.num_points, 15, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.zeros((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        # out_dir = os.path.join(os.getcwd(), "renders")
        # os.makedirs(out_dir, exist_ok=True)
        out_dir = self.output_folder
        
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats,
                self.viewmat,
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                self.tile_bounds,
            )
            torch.cuda.synchronize()

            times[0] += time.time() - start
            start = time.time()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
            )
            torch.cuda.synchronize()


            times[1] += time.time() - start

            # losses for reconstruction
            loss_l1 = l1_loss(out_img, self.gt_image)
            loss_mse = mse_loss(out_img, self.gt_image)
            loss_masked_mse = masked_mse_loss(out_img, self.gt_image)
            loss_masked_l1 = masked_l1_loss(out_img, self.gt_image)
            loss_bg = bg_loss(out_img, self.gt_image)
            loss_ssim = ssim_loss(out_img, self.gt_image)

            # loss for calibration
            loss_cos_dist = codebook_cos_loss(self.rgbs, self.codebook)

            loss = 0

            if self.cfg['w_l1'] > 0:
                loss += self.cfg['w_l1'] * loss_l1
            if self.cfg['w_l2'] > 0:
                loss += self.cfg['w_l2'] * loss_mse
            if self.cfg['w_lml1'] > 0:
                loss += self.cfg['w_lml1'] * loss_masked_l1
            if self.cfg['w_lml2'] > 0:
                loss += self.cfg['w_lml2'] * loss_masked_mse
            if self.cfg['w_bg'] > 0:
                loss += self.cfg['w_bg'] * loss_bg
            if self.cfg['w_ssim'] > 0:
                loss += self.cfg['w_ssim'] * loss_ssim
            if self.cfg['w_code_cos'] > 0:
                loss += self.cfg['w_code_cos'] * loss_cos_dist
                

            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()

            with torch.no_grad():
            
                if save_imgs and iter % 100 == 0:
                    # count psnr for each channel, out_img: [h, w, 15]
                    psnr = []

                    for i in range(15):
                        mse = torch.mean((out_img[..., i] - self.gt_image[..., i]) ** 2).cpu()
                        psnr.append(float(10 * torch.log10(1 / mse)))
                    
                    mean_mse = torch.mean((out_img - self.gt_image) ** 2).cpu()
                    mean_psnr = float(10 * torch.log10(1 / mean_mse))

                    print(f"Iter {iter + 1}/{iterations}, L: {loss:.7f}, Ll2: {loss_mse:.7f}, Lml2: {loss_masked_mse:.7f}, Lssim: {loss_ssim:.7f}, mPSNR: {mean_psnr:.2f}")

                    wandb.log({
                        "loss/total": loss,
                        "loss/l2": loss_mse,
                        "loss/l1": loss_l1,
                        "loss/lml2": loss_masked_mse,
                        "loss/lml1": loss_masked_l1,
                        "loss/bg": loss_bg,
                        "loss/ssim": loss_ssim,
                        "loss/code_cos": loss_cos_dist,
                        "psnr/mean": mean_psnr,
                    }, step=iter)

                    wandb.log({f"psnr/image_{i}": psnr[i] for i in range(15)}, step=iter)

                if save_imgs and iter % 500 == 0:
                    view = view_output(out_img, self.gt_image)
                    frames.append(view)

                    # save last view
                    Image.fromarray(view).save(f"{out_dir}/last.png",)

                    wandb.log({"view/recon": wandb.Image(view)}, step=iter)


        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )

            # save last view
            frames[-1].save(
                f"{out_dir}/output.png",
            )


        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )

        # save csv
        write_to_csv(
            image=self.gt_image[..., 0],
            pixel_coords=xys,
            alpha=self.rgbs,
            save_path=f"{out_dir}/output.csv",
            h=self.H,
            w=self.W,
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]

    # if img_tensor.shape[-1] == 1:
    #     img_tensor = img_tensor.repeat(1, 1, 3)

    return img_tensor


def images_to_tensor(image_path: Path):
    import torchvision.transforms as transforms
    import glob

    # image_paths = [image_path / f'F1R{r}Ch{c}.png' for r in range(1, 6) for c in range(2, 5)]
    image_paths = [image_path / f'{i}.png' for i in range(1, 16)]
    
    images = []

    for image_path in image_paths:
        img = Image.open(image_path)
        transform = transforms.ToTensor()
        img_tensor = transform(img).permute(1, 2, 0)[..., :3]

        images.append(img_tensor)
    
    imgs_tensor = torch.cat(images, dim=2)

    return imgs_tensor


def view_output(pred, gt):
    # view pred and gt images in 15 groups.
    # for each group, pred on left, gt on right, pred: [h, w, 15], gt: [h, w, 15], using heatmap

    import matplotlib.pyplot as plt

    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    fig, axs = plt.subplots(6, 5, figsize=(15, 15))
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0, hspace=0)
    # plt.margins(0, 0)

    for i in range(3):
        for j in range(5):
            # pred
            axs[i * 2 + 1, j].imshow(pred[..., i * 5 + j], cmap='jet', interpolation='nearest')
            axs[i * 2 + 1, j].axis('off')
            axs[i * 2 + 1, j].set_title(f'Pred {i * 5 + j}')

            # gt
            axs[i * 2, j].imshow(gt[..., i * 5 + j], cmap='jet', interpolation='nearest')
            axs[i * 2, j].axis('off')
            axs[i * 2, j].set_title(f'GT {i * 5 + j}')
            
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.15)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()

    return data


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 8000,
    save_imgs: bool = True,
    # img_path: Optional[Path] = None, # 'data/POOL_uint8regi_cropped_0/F1R1Ch2.png',
    img_path: Optional[Path] = Path('data/1213_demo_data_v2/raw1'),
    iterations: int = 20000,
    lr: float = 0.002,
    exp_name: str = 'debug',
    weights: list[float] = [0, 1, 0, 0, 0, 0, 0.001]   # l1, l2, lml1, lml2, bg, ssim, code_cos
) -> None:
    
    config = {
        'w_l1': weights[0],
        'w_l2': weights[1],
        'w_lml1': weights[2],
        'w_lml2': weights[3],
        'w_bg': weights[4],
        'w_ssim': weights[5],
        'w_code_cos': weights[6],
        'run_name': exp_name,
        'exp_name': exp_name,
        'codebook_path': 'data/codebook.xlsx',
    }
    
    wandb.init(
        project="rna_cali",
        config=config,
        name=config["run_name"],
    )

    print(f"Running with config: {config}")

    
    if img_path:
        # gt_image = image_path_to_tensor(img_path)
        gt_image = images_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(
        gt_image=gt_image, 
        num_points=num_points,
        cfg=config,
    )
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)