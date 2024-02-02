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
from gsplat2d import *

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
    obtain_sigma_xy_rho,
    size_loss,
    circle_loss,
    rho_loss,
    mdp_loss,
    r1r2_loss,
)

from utils import (
    write_to_csv,
    write_to_csv_all,
    write_to_csv_hamming,
    read_codebook,
    MDP_recon_psnr,
)

from preprocess import preprocess_data, give_required_data
from visualize import view_output


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor = None,
        primary_samples: int = 20000,
        backup_samples: int = 8000,
        image_file_name: Path = "",
        image_size: list = [4001, 4001, 3],
        densification_interval: int = 1000,
        cfg: Optional[dict] = None,
        model_path: str = None,
        test_points_mode=False,
    ):
        if test_points_mode:
            return
        # (gzx):load model weights to
        if model_path != None:
            self.device = torch.device("cuda:0")

            model = torch.load(model_path, map_location=torch.device(self.device))

            self.begin_iter = model["iters"]
            self.means = model["means"]
            self.rgbs = model["rgbs"]
            self.persistent_mask = model["persistent_mask"]
            self.scales = model["scales"]
            self.quats = model["quats"]
            self.opacities = model["opacities"]
            self.current_marker = model["current_marker"]
            self.H, self.W = model["H"], model["W"]
            self.focal = model["focal"]
            self.tile_bounds = model["tile_bounds"]
            self.cfg = model["cfg"]

            cfg = self.cfg

            self.gt_image = gt_image.to(device=self.device)
            self.num_points = self.means.shape[0]
            # self.image_size = image_size
            self.image_size = self.gt_image.shape
            self.prune_threshold = cfg["prune_threshold"]
            self.grad_threshold = cfg["grad_threshold"]
            self.gauss_threshold = cfg["gauss_threshold"]
            self.prune_flag = cfg["prune_flag"]
            self.split_flag = cfg["split_flag"]
            self.clone_flag = cfg["clone_flag"]
            self.pos_score = cfg["pos_score"]
            self.densification_interval = densification_interval
            self.save_interval = cfg["save_interval"]
            # self.kernal_size = kernal_size

            BLOCK_X, BLOCK_Y = 16, 16
            fov_x = math.pi / 2.0

            self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
            self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

            self.means.requires_grad = True
            self.scales.requires_grad = True
            self.quats.requires_grad = True
            self.rgbs.requires_grad = True
            self.opacities.requires_grad = True

            self.codebook = torch.tensor(
                read_codebook(cfg["codebook_path"]), device=self.device
            )

            self.output_folder = os.path.join("outputs", self.cfg["exp_name"])
            os.makedirs(self.output_folder, exist_ok=True)
            return

        self.begin_iter = 0
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        # self.num_points = num_points
        self.primary_samples = primary_samples
        self.backup_samples = backup_samples
        # self.image_size = image_size
        self.image_size = self.gt_image.shape
        self.prune_threshold = cfg["prune_threshold"]
        self.grad_threshold = cfg["grad_threshold"]
        self.gauss_threshold = cfg["gauss_threshold"]
        self.prune_flag = cfg["prune_flag"]
        self.split_flag = cfg["split_flag"]
        self.clone_flag = cfg["clone_flag"]
        self.initialization = cfg["initialization"]
        self.pos_score = cfg["pos_score"]
        self.densification_interval = densification_interval
        self.save_interval = cfg["save_interval"]
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

        # self.min_sigma = self.depth * 1.0 / self.H
        # self.max_sigma = self.depth * 4.9 / self.H
        self.min_sigma = 1
        self.max_sigma = 5
        print(self.min_sigma, self.max_sigma)

    # (zwx)
    def _init_gaussians(self, image_file_name):
        self.num_points = self.primary_samples + self.backup_samples
        coords = np.random.randint(0, [self.W, self.H], size=(self.num_points, 2))

        if self.initialization:

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

        relative_coords = (coords / [self.H, self.W]) * 2 - 1

        self.xys = torch.from_numpy(relative_coords.astype(np.float32)).to(self.device)

        # scale to absolute coordinates
        self.scale_xy = torch.tensor([self.H, self.W], device=self.device, dtype=torch.float32).reshape(1, 2)

        self.rgbs = (
            torch.ones(self.num_points, self.gt_image.shape[-1], device=self.device)
            # * torch.clamp(colour_values,0.02,0.98)
            # * (colour_values - 1.0)
            * (colour_values)
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

        self.rho = (
            torch.rand(self.num_points, 1, device=self.device) * 1
        )  # (lz) start from 1
        self.sigma_x = torch.ones(self.num_points, 1, device=self.device) * 0.5
        self.sigma_y = torch.ones(self.num_points, 1, device=self.device) * 0.5
        self.opacities = torch.ones((self.num_points, 1), device=self.device) * 1

        self.depth = 30.0

        self.xys.requires_grad = True
        self.rho.requires_grad = True
        self.sigma_x.requires_grad = True
        self.sigma_y.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = False

        # W_values = torch.cat([self.scales, self.quats, self.rgbs, self.means], dim=1)

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        # out_dir = os.path.join(os.getcwd(), "renders")
        # os.makedirs(out_dir, exist_ok=True)
        out_dir = self.output_folder

        optimizer = optim.Adam(
            # [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
            [self.rgbs, self.xys, self.sigma_x, self.sigma_y, self.rho],
            lr,  # (lz) do not update opacities
        )
        # start0 = time.time() #(gzx):计算时间
        frames = []
        times = [0] * 3  # project, rasterize, backward
        formal_code_loss = 0
        for iter in range(self.begin_iter, iterations):

            # (gzx)
            if (
                iter % (self.densification_interval + 1) == 0
                and iter > 0
                and self.prune_flag
            ):
                indices_to_remove = (
                    torch.sigmoid(self.rgbs).max(dim=-1)[0] < self.prune_threshold
                ).nonzero(as_tuple=True)[0]

                if len(indices_to_remove) > 0:
                    print(f"number of pruned points: {len(indices_to_remove)}")

                self.persistent_mask[indices_to_remove] = False

                # Zero-out parameters and their gradients at every epoch using the persistent mask
                self.means.data[~self.persistent_mask] = 0.0
                self.scales.data[~self.persistent_mask] = 0.0
                self.quats.data[~self.persistent_mask] = 0.0
                self.rgbs.data[~self.persistent_mask] = 0.0
            # (gzx):这条指令好像很拖速度
            # gc.collect()

            xys, rho, sigma_x, sigma_y, persist_rgbs, persist_opacities = self.get_persist()

            depths, radii, conics, num_tiles_hit = project_gaussians_2D(
                sigma_x, sigma_y, rho, self.xys.shape[0], self.depth, device=self.device
            )

            if iter == 0:
                for i in range(10):
                    print(sigma_x[i], sigma_y[i], rho[i], radii[i])

            torch.cuda.synchronize()

            start = time.time()
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()

            absoluate_xys = self.relative_to_absolute_coords(xys)

            out_img = rasterize_gaussians(
                absoluate_xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                persist_rgbs,
                persist_opacities,
                self.H,
                self.W,
            )
            torch.cuda.synchronize()

            times[1] += time.time() - start

            # (zwx) loss for calibration
            flag = True
            alpha = persist_rgbs * persist_opacities
            # assert(persist_rgbs <= 1 and persist_rgbs >=0)
            # assert(persist_opacities <= 1 and persist_opacities >=0)
            loss = 0

            # (lz) count loss required for training
            if self.cfg["w_size"] > 0 or self.cfg["w_rho"] > 0:
                lambda1, lambda2 = get_lambda(sigma_x, sigma_y, rho)
                r1, r2 = (torch.sqrt(3 * lambda1)), (torch.sqrt(3 * lambda2))

            if self.cfg["w_mdp"] > 0:  # (zwx)
                loss_mdp = mdp_loss(out_img, self.gt_image)
                loss += self.cfg["w_mdp"] * loss_mdp
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
                loss_size = size_loss(
                    r1,
                    r2,
                    min_size=self.cfg["size_range"][0],
                    max_size=self.cfg["size_range"][1],
                )
                loss += self.cfg["w_size"] * loss_size

            if self.cfg["w_code"] > 0:
                loss_cos_dist, flag, formal_code_loss = codebook_loss(
                    self.cfg["cali_loss_type"],
                    alpha,
                    self.codebook,
                    flag,
                    iter,
                    formal_code_loss,
                )
                loss += self.cfg["w_code"] * loss_cos_dist

            if self.cfg["w_rho"] > 0:
                loss_rho = r1r2_loss(r2 / r1)
                loss += self.cfg["w_rho"] * loss_rho

            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()

            # (gzx)
            if (
                iter % self.densification_interval == 0
                and iter > 0
                and (self.split_flag or self.clone_flag)
            ):
                # Calculate the norm of gradients
                gradient_norms = torch.norm(
                    self.means.grad[self.persistent_mask], dim=1, p=2
                )
                gaussian_norms = torch.norm(
                    torch.sigmoid(self.scales.data[self.persistent_mask]), dim=1, p=2
                )

                sorted_grads, sorted_grads_indices = torch.sort(
                    gradient_norms, descending=True
                )
                sorted_gauss, sorted_gauss_indices = torch.sort(
                    gaussian_norms, descending=True
                )

                large_gradient_mask = sorted_grads > self.grad_threshold
                large_gradient_indices = sorted_grads_indices[large_gradient_mask]

                large_gauss_mask = sorted_gauss > self.gauss_threshold
                large_gauss_indices = sorted_gauss_indices[large_gauss_mask]

                common_indices_mask = torch.isin(
                    large_gradient_indices, large_gauss_indices
                )
                common_indices = large_gradient_indices[common_indices_mask]
                distinct_indices = large_gradient_indices[~common_indices_mask]

                # Split points with large coordinate gradient and large gaussian values and descale their gaussian
                if len(common_indices) > 0 and self.split_flag:
                    print(f"number of splitted points: {len(common_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(common_indices)
                    self.persistent_mask[start_index:end_index] = True
                    self.means.data[start_index:end_index, :] = self.means.data[
                        common_indices, :
                    ]  # (zwx) mean -> means
                    self.scales.data[start_index:end_index, :] = self.scales.data[
                        common_indices, :
                    ]
                    self.quats.data[start_index:end_index, :] = self.quats.data[
                        common_indices, :
                    ]
                    self.rgbs.data[start_index:end_index, :] = self.rgbs.data[
                        common_indices, :
                    ]

                    scale_reduction_factor = 1.6
                    self.scales.data[start_index:end_index] /= scale_reduction_factor
                    self.scales.data[common_indices] /= scale_reduction_factor
                    self.current_marker = self.current_marker + len(common_indices)

                # Clone it points with large coordinate gradient and small gaussian values
                if len(distinct_indices) > 0 and self.clone_flag:
                    print(f"number of cloned points: {len(distinct_indices)}")
                    start_index = self.current_marker + 1
                    end_index = self.current_marker + 1 + len(distinct_indices)
                    self.persistent_mask[start_index:end_index] = True
                    self.means.data[start_index:end_index, :] = self.means.data[
                        distinct_indices, :
                    ]
                    self.scales.data[start_index:end_index, :] = self.scales.data[
                        distinct_indices, :
                    ]
                    self.quats.data[start_index:end_index, :] = self.quats.data[
                        distinct_indices, :
                    ]
                    self.rgbs.data[start_index:end_index, :] = self.rgbs.data[
                        distinct_indices, :
                    ]
                    current_marker = current_marker + len(distinct_indices)

            times[2] += time.time() - start
            optimizer.step()

            self.validation(
                save_imgs=save_imgs,
                loss=loss,
                out_img=out_img,
                persist_rgbs=persist_rgbs,
                conics=conics,
                alpha=persist_rgbs,
                iter=iter,
                iterations=iterations,
                frames=frames,
                out_dir=out_dir,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                rho=rho,
            )

            if (iter) % 500 == 0:
                write_to_csv_all(
                    pixel_coords=xys,
                    xyr=[sigma_x, sigma_y, rho],
                    alpha=alpha,
                    save_path=f"{out_dir}/output_all.csv",
                    h=self.H,
                    w=self.W,
                    ref=self.gt_image,
                    codebook_path=self.cfg["codebook_path"],
                )

            if (iter + 1) % self.save_interval == 0:
                torch.save(
                    {
                        "iters": iter + 1,
                        "xys": self.xys,
                        "rho": self.rho,
                        "sigma_x": self.sigma_x,
                        "sigma_y": self.sigma_y,
                        "depth": self.depth,
                        "rgbs": self.rgbs,
                        "opacities": self.opacities,
                        "persistent_mask": self.persistent_mask,
                        "focal": self.focal,
                        "H": self.H,
                        "W": self.W,
                        "tile_bounds": self.tile_bounds,
                        "current_marker": self.current_marker,
                        "cfg": self.cfg,
                    },
                    os.path.join(out_dir, f"params_{iter+1}.pth"),
                )

        # cyy: add what project need (TODO: fix bugs in self.)
        torch.save(
            {
                "iters": iter + 1,
                "xys": self.xys,
                "rho": self.rho,
                "sigma_x": self.sigma_x,
                "sigma_y": self.sigma_y,
                "depth": self.depth,
                "rgbs": self.rgbs,
                "opacities": self.opacities,
                "persistent_mask": self.persistent_mask,
                "focal": self.focal,
                "H": self.H,
                "W": self.W,
                "tile_bounds": self.tile_bounds,
                "current_marker": self.current_marker,
                "cfg": self.cfg,
            },
            os.path.join(out_dir, "params.pth"),
        )

        # (zwx) print code loss
        print(
            "test_li:",
            li_codeloss(persist_rgbs, self.codebook)[0].item(),
            "test_otsu:",
            otsu_codeloss(persist_rgbs, self.codebook)[0].item(),
            "test_hamming_normal:",
            codebook_hamming_loss(persist_rgbs, self.codebook, "normal")[0].item(),
            "test_hamming_mean:",
            codebook_hamming_loss(persist_rgbs, self.codebook, "mean")[0].item(),
            "test_hamming_median:",
            codebook_hamming_loss(persist_rgbs, self.codebook, "median")[0].item(),
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

        # cyy ：增加训练结束时写所有距离
        write_to_csv_all(
            pixel_coords=xys,
            xyr=[sigma_x, sigma_y, rho],
            alpha=alpha,
            save_path=f"{out_dir}/output_all.csv",
            h=self.H,
            w=self.W,
            ref=self.gt_image,
            codebook_path=self.cfg["codebook_path"],
        )

    # (lz) separate validation function
    @torch.no_grad()
    def validation(
        self,
        save_imgs,
        loss,
        out_img,
        persist_rgbs,
        conics,
        alpha,
        iter,
        iterations,
        frames,
        out_dir,
        sigma_x,
        sigma_y,
        rho,
    ):
        if save_imgs and iter % 100 == 0:
            # count loss for validation
            # sigma_x, sigma_y = obtain_sigma_xy(conics)
            # sigma_x, sigma_y, rho = obtain_sigma_xy_rho(conics)

            loss_size = size_loss(sigma_x, sigma_y, min_size=6, max_size=12)
            loss_circle = circle_loss(sigma_x, sigma_y)
            loss_rho = rho_loss(sigma_x, sigma_y, rho)

            loss_l1 = l1_loss(out_img, self.gt_image)
            loss_mse = mse_loss(out_img, self.gt_image)
            loss_masked_l1 = masked_l1_loss(out_img, self.gt_image)
            loss_masked_mse = masked_mse_loss(out_img, self.gt_image)
            loss_bg = bg_loss(out_img, self.gt_image)
            loss_ssim = ssim_loss(out_img, self.gt_image)

            loss_cos_dist = codebook_cos_loss(alpha, self.codebook)
            loss_nml_hm_dist, _ = codebook_hamming_loss(alpha, self.codebook, "normal")
            loss_mean_hm_dist, _ = codebook_hamming_loss(alpha, self.codebook, "mean")
            loss_median_hm_dist, _ = codebook_hamming_loss(
                alpha, self.codebook, "median"
            )
            loss_li_hm_dist, _ = li_codeloss(alpha, self.codebook)
            loss_otsu_hm_dist, _ = otsu_codeloss(alpha, self.codebook)

            loss_mdp = mdp_loss(out_img, self.gt_image)

            # count psnr for each channel, out_img: [h, w, 15]
            psnr = []

            for i in range(15):
                mse = torch.mean((out_img[..., i] - self.gt_image[..., i]) ** 2).cpu()
                psnr.append(float(10 * torch.log10(1 / mse)))

            mean_mse = torch.mean((out_img - self.gt_image) ** 2).cpu()
            mean_psnr = float(10 * torch.log10(1 / mean_mse))
            # (zwx) mpd_psnr
            mdp_psnr = MDP_recon_psnr(out_img, self.gt_image)
            print(
                f"Iter {iter + 1}/{iterations}, N:{persist_rgbs.shape[0]}, Ll2: {loss_mse:.7f}, Lml2: {loss_masked_mse:.7f}, Lssim: {loss_ssim:.7f}, mPSNR: {mean_psnr:.2f}, mdpPSNR: {mdp_psnr:.2f}",
                flush=True,
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
                    "loss/circle_loss": loss_circle,
                    "loss/size_loss": loss_size,
                    "loss/rho_loss": loss_rho,
                    "psnr/mean": mean_psnr,
                    "MDPpsnr": mdp_psnr,
                    "dist/code_cos_loss": loss_cos_dist,
                    "dist/hm_nml_loss": loss_nml_hm_dist,
                    "dist/hm_mean_loss": loss_mean_hm_dist,
                    "dist/hm_median_loss": loss_median_hm_dist,
                    "dist/hm_li_loss": loss_li_hm_dist,
                    "dist/hm_otsu_loss": loss_otsu_hm_dist,
                    "loss/mdp_loss": loss_mdp,
                },
                step=iter,
            )

            wandb.log({f"psnr/image_{i}": psnr[i] for i in range(15)}, step=iter)

        if save_imgs and iter % 500 == 0:
            view = view_output(out_img, self.gt_image)
            frames.append(view)

            # save last view
            Image.fromarray(view).save(
                f"{out_dir}/last.png",
            )

            wandb.log({"view/recon": wandb.Image(view)}, step=iter)

    def get_persist(self):
        xys = self.xys[self.persistent_mask]
        rho = self.rho[self.persistent_mask]
        sigma_x = self.sigma_x[self.persistent_mask]
        sigma_y = self.sigma_y[self.persistent_mask]
        persist_rgbs = torch.sigmoid(self.rgbs[self.persistent_mask])
        persist_opacities = torch.sigmoid(self.opacities[self.persistent_mask])

        return xys, rho, sigma_x, sigma_y, persist_rgbs, persist_opacities

    def relative_to_absolute_coords(self, xys):
        return (xys + 1) / 2 * self.scale_xy

    # (gzx)
    def test(self, model_path):
        model = torch.load(model_path, map_location=torch.device(self.device))

        self.xys = model["xys"]
        self.rgbs = model["rgbs"]
        self.persistent_mask = model["persistent_mask"]
        self.opacities = model["opacities"]
        self.depth = model["depth"]
        self.sigma_x = model["sigma_x"]
        self.sigma_y = model["sigma_y"]
        self.rho = model["rho"]

        xys, rho, sigma_x, sigma_y, persist_rgbs, persist_opacities = self.get_persist()

        depths, radii, conics, num_tiles_hit = project_gaussians_2D(
            sigma_x, sigma_y, rho, self.xys.shape[0], self.depth, device=self.device
        )

        torch.cuda.synchronize()
        out_img = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            persist_rgbs,
            persist_opacities,
            self.H,
            self.W,
        )
        torch.cuda.synchronize()

        alpha = persist_rgbs * persist_opacities
        view = view_output(out_img, self.gt_image)

        Image.fromarray(view).save(
            model_path.replace(".pth", "_out.png"),
        )

        if self.cfg["cali_loss_type"] == "cos":
            write_to_csv(
                image=self.gt_image[..., 0],
                pixel_coords=xys,
                alpha=alpha,
                save_path=model_path.replace(".pth", "_out.csv"),
                h=self.H,
                w=self.W,
                ref=self.gt_image,
                post_processing=self.pos_score,
                pos_threshold=20,
                codebook_path=self.cfg["codebook_path"],
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
                alpha=persist_rgbs,  # (zwx) self.rgbs -> persist_rgbs
                save_path=model_path.replace(".pth", "_out.csv"),
                h=self.H,
                w=self.W,
                ref=self.gt_image,
                post_processing=self.pos_score,
                pos_threshold=20,
                loss=self.cfg["cali_loss_type"],
            )
