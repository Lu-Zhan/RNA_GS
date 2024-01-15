import math
import os
import gc
import time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import torch
import tyro
import cv2
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
import wandb

# (zwx) wandb init
os.environ["WANDB_API_KEY"] = "b3a9139b4f82902def9e2675e768ce664219c4ab"
os.environ["WANDB_MODE"] = "offline"

from losses import (
    mse_loss,
    l1_loss,
    masked_mse_loss,
    masked_l1_loss,
    bg_loss,
    ssim_loss,
    codebook_cos_loss,
)
from utils import write_to_csv, read_codebook


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
        #(gzx):用腐蚀之后的图像
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


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 8000,
        primary_samples: int = 20000,
        backup_samples: int = 8000,
        image_file_name: Path = "",
        image_size: list = [401, 401, 3],
        # kernal_size: int = 25,
        densification_interval:int = 1000,
        cfg: Optional[dict] = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.primary_samples = primary_samples
        self.backup_samples = backup_samples
        self.image_size = image_size
        self.prune_threshold = cfg["prune_threshold"]
        self.grad_threshold =  cfg["grad_threshold"]
        self.gauss_threshold = cfg["gauss_threshold"]
        self.densification_interval = densification_interval
        # self.kernal_size = kernal_size

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

        self._init_gaussians(image_file_name)
        self.codebook = torch.tensor(
            read_codebook(cfg["codebook_path"]), device=self.device
        )
        self.cfg = cfg

        self.output_folder = os.path.join("outputs", self.cfg["exp_name"])
        os.makedirs(self.output_folder, exist_ok=True)

    # (zwx)
    def _init_gaussians(self, image_file_name):
        self.num_points = self.primary_samples + self.backup_samples
        coords = np.random.randint(0, [self.W, self.H], size=(self.num_points, 2))
        num_samples1, coords1 = preprocess_data(Path(image_file_name))
        # padding = self.kernal_size // 2
        if num_samples1 >= self.primary_samples:
            print("too many initial points")
            exit(0)
        if num_samples1 >= self.primary_samples / 2:
            coords_noise = np.random.randint(
                -1, [2, 2], size=(self.primary_samples - num_samples1, 2)
            )
            coords[0:num_samples1, :] = coords1
            coords[num_samples1 : self.primary_samples, :] = np.clip(
                coords1[0 : self.primary_samples - num_samples1, :] + coords_noise,
                0,
                [self.W - 1, self.H - 1],
            )
        else:
            coords_noise = np.random.randint(-1, [2, 2], size=(num_samples1, 2))
            coords[0:num_samples1, :] = coords1
            coords[num_samples1 : num_samples1 * 2, :] = np.clip(
                coords1 + coords_noise, 0, [self.W - 1, self.H - 1]
            )
            self.primary_samples = num_samples1 * 2
            self.backup_samples = self.num_points - self.primary_samples
        colour_values, pixel_coords = give_required_data(
            coords, self.image_size, self.gt_image, self.device
        )
        z = 2 * (torch.rand(self.num_points, 1, device=self.device) - 0.5)
        self.means = torch.cat([pixel_coords, z], dim=1)
        self.rgbs = (
            torch.ones(self.num_points, self.gt_image.shape[-1], device=self.device)
            * colour_values
        )
        starting_size = self.primary_samples
        left_over_size = self.backup_samples
        self.persistent_mask = torch.cat(
            [
                torch.ones(starting_size, dtype=bool),
                torch.zeros(left_over_size, dtype=bool),
            ],
            dim=0,
        )
        self.current_marker = starting_size
        self.scales = torch.rand(self.num_points, 3, device=self.device)

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
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

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
        self.opacities.requires_grad = False
        self.viewmat.requires_grad = False

        # W_values = torch.cat([self.scales, self.quats, self.rgbs, self.means], dim=1)

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        # out_dir = os.path.join(os.getcwd(), "renders")
        # os.makedirs(out_dir, exist_ok=True)
        out_dir = self.output_folder

        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        # start0 = time.time() #(gzx):计算时间
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            
            #(gzx)
            if iter % (self.densification_interval + 1) == 0 and iter > 0:
                indices_to_remove = (torch.sigmoid(self.rgbs).max(dim=-1)[0]  < self.prune_threshold).nonzero(as_tuple=True)[0]

                if len(indices_to_remove) > 0:
                    print(f"number of pruned points: {len(indices_to_remove)}")

                self.persistent_mask[indices_to_remove] = False

                # Zero-out parameters and their gradients at every epoch using the persistent mask
                self.means.data [~self.persistent_mask] = 0.0
                self.scales.data[~self.persistent_mask] = 0.0
                self.quats.data [~self.persistent_mask] = 0.0
                self.rgbs.data  [~self.persistent_mask] = 0.0
            #(gzx):这条指令好像很拖速度
            # gc.collect()
            
            persist_means  = self.means [self.persistent_mask]
            persist_scales = self.scales[self.persistent_mask]
            persist_quats  = self.quats [self.persistent_mask]
            persist_rgbs   = self.rgbs  [self.persistent_mask]


            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
                persist_means,
                persist_scales,
                1,
                persist_quats,
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
                torch.sigmoid(persist_rgbs),
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
            loss_cos_dist = codebook_cos_loss(persist_rgbs, self.codebook)

            loss = 0

            if self.cfg["w_l1"] > 0:
                loss += self.cfg["w_l1"] * loss_l1
            if self.cfg["w_l2"] > 0:
                loss += self.cfg["w_l2"] * loss_mse
            if self.cfg["w_lml1"] > 0:
                loss += self.cfg["w_lml1"] * loss_masked_l1
            if self.cfg["w_lml2"] > 0:
                loss += self.cfg["w_lml2"] * loss_masked_mse
            if self.cfg["w_bg"] > 0:
                loss += self.cfg["w_bg"] * loss_bg
            if self.cfg["w_ssim"] > 0:
                loss += self.cfg["w_ssim"] * loss_ssim
            if self.cfg["w_code_cos"] > 0:
                loss += self.cfg["w_code_cos"] * loss_cos_dist

            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            
            #(gzx):在反向传播前将梯度置0
            if self.persistent_mask is not None:
                self.means.grad.data [~self.persistent_mask] = 0.0
                self.scales.grad.data[~self.persistent_mask] = 0.0
                self.quats.grad.data [~self.persistent_mask] = 0.0
                self.rgbs.grad.data  [~self.persistent_mask] = 0.0  
            #(gzx)
            if iter % self.densification_interval == 0 and iter > 0:
                # Calculate the norm of gradients
                gradient_norms = torch.norm(self.means.grad[self.persistent_mask], dim=1, p=2)
                gaussian_norms = torch.norm(torch.sigmoid(self.scales.data[self.persistent_mask]), dim=1, p=2)

                sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
                sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)

                large_gradient_mask = (sorted_grads > self.grad_threshold)
                large_gradient_indices = sorted_grads_indices[large_gradient_mask]

                large_gauss_mask = (sorted_gauss > self.gauss_threshold)
                large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

                common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
                common_indices = large_gradient_indices[common_indices_mask]
                distinct_indices = large_gradient_indices[~common_indices_mask]

                # Split points with large coordinate gradient and large gaussian values and descale their gaussian
                if len(common_indices) > 0:
                    print(f"number of splitted points: {len(common_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(common_indices)
                    self.persistent_mask[start_index: end_index] = True
                    self.mean.data[start_index:end_index, :]   = self.mean.data[common_indices, :]
                    self.scales.data[start_index:end_index, :] = self.scales.data[common_indices, :]
                    self.quats.data[start_index:end_index, :]  = self.quats.data[common_indices, :]
                    self.rgbs.data[start_index:end_index, :]   = self.rgbs.data[common_indices, :]  

                    scale_reduction_factor = 1.6
                    self.scales.data[start_index:end_index] /= scale_reduction_factor
                    self.scales.data[common_indices] /= scale_reduction_factor
                    self.current_marker = self.current_marker + len(common_indices)

                # Clone it points with large coordinate gradient and small gaussian values
                if len(distinct_indices) > 0:
                    print(f"number of cloned points: {len(distinct_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(distinct_indices)
                    self.persistent_mask[start_index: end_index] = True
                    self.mean.data[start_index:end_index, :]   = self.mean.data[distinct_indices, :]
                    self.scales.data[start_index:end_index, :] = self.scales.data[distinct_indices, :]
                    self.quats.data[start_index:end_index, :]  = self.quats.data[distinct_indices, :]
                    self.rgbs.data[start_index:end_index, :]   = self.rgbs.data[distinct_indices, :]  
                    current_marker = current_marker + len(distinct_indices)

            

            
            times[2] += time.time() - start
            optimizer.step()

            with torch.no_grad():
                if save_imgs and iter % 100 == 0:
                    # count psnr for each channel, out_img: [h, w, 15]
                    psnr = []

                    for i in range(15):
                        mse = torch.mean(
                            (out_img[..., i] - self.gt_image[..., i]) ** 2
                        ).cpu()
                        psnr.append(float(10 * torch.log10(1 / mse)))

                    mean_mse = torch.mean((out_img - self.gt_image) ** 2).cpu()
                    mean_psnr = float(10 * torch.log10(1 / mean_mse))

                    #(gzx):检查时间
                    # if (iter%self.densification_interval==0):
                    #     print(f"Iter1-{iter+1} use{(time.time()-start0):.2f}s.")
                    
                    print(
                        f"Iter {iter + 1}/{iterations}, N:{persist_rgbs.shape[0]}, L: {loss:.7f}, Ll2: {loss_mse:.7f}, Lml2: {loss_masked_mse:.7f}, Lssim: {loss_ssim:.7f}, mPSNR: {mean_psnr:.2f}"
                    )

                    wandb.log(
                        {
                            "point_number": persist_rgbs.shape[0],
                            "loss/total": loss,
                            "loss/l2": loss_mse,
                            "loss/l1": loss_l1,
                            "loss/lml2": loss_masked_mse,
                            "loss/lml1": loss_masked_l1,
                            "loss/bg": loss_bg,
                            "loss/ssim": loss_ssim,
                            "loss/code_cos": loss_cos_dist,
                            "psnr/mean": mean_psnr,
                        },
                        step=iter,
                    )

                    wandb.log(
                        {f"psnr/image_{i}": psnr[i] for i in range(15)}, step=iter
                    )

                if save_imgs and iter % 500 == 0:
                    view = view_output(out_img, self.gt_image)
                    frames.append(view)

                    # save last view
                    Image.fromarray(view).save(
                        f"{out_dir}/last.png",
                    )

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
            alpha=persist_rgbs,
            save_path=f"{out_dir}/output.csv",
            h=self.H,
            w=self.W,
            codebook_path = self.cfg["codebook_path"],
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
    image_paths = [image_path / f"{i}.png" for i in range(1, 16)]

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


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 8000,
    save_imgs: bool = True,
    # img_path: Optional[Path] = None, # 'data/POOL_uint8regi_cropped_0/F1R1Ch2.png',
    img_path: Optional[Path] = Path("./data/1213_demo_data_v2/raw1"),
    codebook_path: Optional[Path] = Path("data/codebook.xlsx"),
    
    iterations: int = 20000,
    densification_interval: int = 1000,
    lr: float = 0.002,
    exp_name: str = "debug",
    weights: list[float] = [
        0,
        1,
        0,
        0,
        0,
        0,
        0.001,
    ],  # l1, l2, lml1, lml2, bg, ssim, code_cos
    thresholds: list[float] = [
        0.3,
        0.04,
        0.04,
    ],  # l1, l2, lml1, lml2, bg, ssim, code_cos
) -> None:
    config = {
        "w_l1": weights[0],
        "w_l2": weights[1],
        "w_lml1": weights[2],
        "w_lml2": weights[3],
        "w_bg": weights[4],
        "w_ssim": weights[5],
        "w_code_cos": weights[6],
        "prune_threshold" : thresholds[0],
        "grad_threshold" : thresholds[1],
        "gauss_threshold" : thresholds[2],
        "exp_name": exp_name,
        "codebook_path": codebook_path,
    }

    wandb.init(
        project="rna_cali",
        config=config,
        name=config["exp_name"],
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
        image_file_name=img_path,
        densification_interval=densification_interval,
    )
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)