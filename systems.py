import math
import os
import gc
import time
import torch
import wandb
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional
from torch import Tensor, optim
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from losses import (
    mse_loss,
    l1_loss,
    masked_mse_loss,
    masked_l1_loss,
    bg_loss,
    ssim_loss,
    codebook_loss,
    codebook_cos_loss,
    li_codeloss,
    otsu_codeloss,
    codebook_hamming_loss,
    obtain_sigma_xy,
    size_loss,
    circle_loss,
    obtain_sigma_xy,
    size_loss,
    circle_loss,
    obtain_sigma_xy,
    size_loss,
    circle_loss,
    obtain_sigma_xy,
    size_loss,
    circle_loss,
)

from utils import (
    write_to_csv,
    write_to_csv_hamming,
    read_codebook,
)

from preprocess import preprocess_data, give_required_data
from visualize import view_output


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        primary_samples: int = 20000,
        backup_samples: int = 8000,
        image_file_name: Path = "",
        image_size: list = [401, 401, 3],
        densification_interval: int = 1000,
        cfg: Optional[dict] = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        # self.num_points = num_points
        self.primary_samples = primary_samples
        self.backup_samples = backup_samples
        self.image_size = image_size
        self.prune_threshold = cfg["prune_threshold"]
        self.grad_threshold = cfg["grad_threshold"]
        self.gauss_threshold = cfg["gauss_threshold"]
        self.prune_flag = cfg["prune_flag"]
        self.split_flag = cfg["split_flag"]
        self.clone_flag = cfg["clone_flag"]
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
            if iter % (self.densification_interval + 1) == 0 and iter > 0 and self.prune_flag:
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

            # (zwx) loss for calibration
            flag = True
            alpha = torch.sigmoid(persist_rgbs)
            loss = 0

            # (lz) count loss required for training
            if self.cfg["w_circle"] > 0 or self.cfg["w_size"] > 0:
                sigma_x, sigma_y = obtain_sigma_xy(conics)

            if self.cfg["w_l1"] > 0:
                loss_l1 = l1_loss(out_img, self.gt_image)
                loss += self.cfg["w_l1"] * loss_l1
            if self.cfg["w_l2"] > 0:
                loss_mse = mse_loss(out_img, self.gt_image)
                loss += self.cfg["w_l2"] * loss_mse
            if self.cfg["w_lml1"] > 0:
                loss_masked_l1 = masked_l1_loss(out_img, self.gt_image)
                loss += self.cfg["w_lml1"] * loss_masked_l1
            if self.cfg["w_lml2"] > 0:
                loss_masked_mse = masked_mse_loss(out_img, self.gt_image)
                loss += self.cfg["w_lml2"] * loss_masked_mse
            if self.cfg["w_bg"] > 0:
                loss_bg = bg_loss(out_img, self.gt_image)
                loss += self.cfg["w_bg"] * loss_bg
            if self.cfg["w_ssim"] > 0:
                loss_ssim = ssim_loss(out_img, self.gt_image)
                loss += self.cfg["w_ssim"] * loss_ssim
            if self.cfg["w_circle"] > 0:
                loss_circle = circle_loss(sigma_x, sigma_y)
                loss += self.cfg["w_circle"] * loss_circle
            if self.cfg["w_size"] > 0:
                loss_size = size_loss(sigma_x, sigma_y, min_size=6, max_size=12)
                loss += self.cfg["w_size"] * loss_size
            if self.cfg["w_code_cos"] > 0:
                loss_cos_dist, flag = codebook_loss(self.cfg["cali_loss_type"], alpha, self.codebook, flag, iter)
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
            if iter % self.densification_interval == 0 and iter > 0 and (self.split_flag or self.clone_flag):
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
                if len(common_indices) > 0 and self.split_flag:
                    print(f"number of splitted points: {len(common_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(common_indices)
                    self.persistent_mask[start_index: end_index] = True
                    self.means.data[start_index:end_index, :]  = self.means.data[common_indices, :]  # (zwx) mean -> means
                    self.scales.data[start_index:end_index, :] = self.scales.data[common_indices, :]
                    self.quats.data[start_index:end_index, :]  = self.quats.data[common_indices, :]
                    self.rgbs.data[start_index:end_index, :]   = self.rgbs.data[common_indices, :]  

                    scale_reduction_factor = 1.6
                    self.scales.data[start_index:end_index] /= scale_reduction_factor
                    self.scales.data[common_indices] /= scale_reduction_factor
                    self.current_marker = self.current_marker + len(common_indices)

                # Clone it points with large coordinate gradient and small gaussian values
                if len(distinct_indices) > 0 and self.clone_flag:
                    print(f"number of cloned points: {len(distinct_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(distinct_indices)
                    self.persistent_mask[start_index: end_index] = True
                    self.means.data[start_index:end_index, :]  = self.means.data[distinct_indices, :]
                    self.scales.data[start_index:end_index, :] = self.scales.data[distinct_indices, :]
                    self.quats.data[start_index:end_index, :]  = self.quats.data[distinct_indices, :]
                    self.rgbs.data[start_index:end_index, :]   = self.rgbs.data[distinct_indices, :]  
                    current_marker = current_marker + len(distinct_indices)
            
            times[2] += time.time() - start
            optimizer.step()

            self.validation(
                save_imgs=save_imgs, 
                loss=loss, 
                out_img=out_img, 
                persist_rgbs=persist_rgbs, 
                conics=conics, 
                alpha=torch.sigmoid(persist_rgbs), 
                iter=iter, 
                iterations=iterations, 
                frames=frames, 
                out_dir=out_dir,
            )
        
        # (zwx) print code loss
        print("test_li:", li_codeloss(torch.sigmoid(persist_rgbs), self.codebook)[0].item(),
              "test_otsu:", otsu_codeloss(torch.sigmoid(persist_rgbs), self.codebook)[0].item(),
              "test_hamming_normal:", codebook_hamming_loss(torch.sigmoid(persist_rgbs), self.codebook, "normal")[0].item(),
              "test_hamming_mean:", codebook_hamming_loss(torch.sigmoid(persist_rgbs), self.codebook, "mean")[0].item(),
              "test_hamming_median:", codebook_hamming_loss(torch.sigmoid(persist_rgbs), self.codebook, "median")[0].item()
        )
        
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

        # (zwx) save csv
        if self.cfg["cali_loss_type"] == "cos":
            write_to_csv(
                image=self.gt_image[..., 0],
                pixel_coords=xys,
                alpha=persist_rgbs,
                save_path=f"{out_dir}/output.csv",
                h=self.H,
                w=self.W,
                codebook_path = self.cfg["codebook_path"],
        )
        elif (
            self.cfg["cali_loss_type"] == "mean"
            or self.cfg["cali_loss_type"] == "median"
            or self.cfg["cali_loss_type"] == "li"
            or self.cfg["cali_loss_type"] == "otsu"
        ):
            write_to_csv_hamming(
                image=self.gt_image[..., 0],
                pixel_coords=xys,
                alpha=persist_rgbs, # (zwx) self.rgbs -> persist_rgbs
                save_path=f"{out_dir}/output.csv",
                h=self.H,
                w=self.W,
                loss=self.cfg["cali_loss_type"],
            )

    # (lz) separate validation function
    @torch.no_grad()
    def validation(self, save_imgs, loss, out_img, persist_rgbs, conics, alpha, iter, iterations, frames, out_dir):
        if save_imgs and iter % 100 == 0:
            # count loss for validation
            sigma_x, sigma_y = obtain_sigma_xy(conics)
            loss_size = size_loss(sigma_x, sigma_y, min_size=6, max_size=12)
            loss_circle = circle_loss(sigma_x, sigma_y)

            loss_l1 = l1_loss(out_img, self.gt_image)
            loss_mse = mse_loss(out_img, self.gt_image)
            loss_masked_l1 = masked_l1_loss(out_img, self.gt_image)
            loss_masked_mse = masked_mse_loss(out_img, self.gt_image)
            loss_bg = bg_loss(out_img, self.gt_image)
            loss_ssim = ssim_loss(out_img, self.gt_image)

            loss_cos_dist = codebook_cos_loss(alpha, self.codebook)
            loss_nml_hm_dist, _ = codebook_hamming_loss(alpha, self.codebook, "normal")
            loss_mean_hm_dist, _ = codebook_hamming_loss(alpha, self.codebook, "mean")
            loss_median_hm_dist, _ = codebook_hamming_loss(alpha, self.codebook, "median")
            loss_li_hm_dist, _ = li_codeloss(alpha, self.codebook)
            loss_otsu_hm_dist, _ = otsu_codeloss(alpha, self.codebook)

            # count psnr for each channel, out_img: [h, w, 15]
            psnr = []

            for i in range(15):
                mse = torch.mean(
                    (out_img[..., i] - self.gt_image[..., i]) ** 2
                ).cpu()
                psnr.append(float(10 * torch.log10(1 / mse)))

            mean_mse = torch.mean((out_img - self.gt_image) ** 2).cpu()
            mean_psnr = float(10 * torch.log10(1 / mean_mse))

            print(
                f"Iter {iter + 1}/{iterations}, N:{persist_rgbs.shape[0]}, Ll2: {loss_mse:.7f}, Lml2: {loss_masked_mse:.7f}, Lssim: {loss_ssim:.7f}, mPSNR: {mean_psnr:.2f}"
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
                    "loss/loss_circle": loss_circle,
                    "loss/size_loss": loss_size,
                    "psnr/mean": mean_psnr,
                    "dist/loss_nml_hm_dist": loss_nml_hm_dist,
                    "dist/loss_mean_hm_dist": loss_mean_hm_dist,
                    "dist/loss_median_hm_dist": loss_median_hm_dist,
                    "dist/loss_li_hm_dist": loss_li_hm_dist,
                    "dist/loss_otsu_hm_dist": loss_otsu_hm_dist,
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


        