import os
import math
import torch

from utils import obtain_init_color, filter_by_background, write_to_csv
from visualize import view_positions
from losses import obtain_simi

class GaussModel(torch.nn.Module):
    def __init__(self, num_primarys, num_backups, hw, device):
        super(GaussModel, self).__init__()
        self._init_gaussians(num_primarys, num_backups, device)

        self._init_mask(num_primarys, num_backups)

        fov_x = math.pi / 2.0
        self.H, self.W = hw[0], hw[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=device)
    
    @property
    def parameters(self):
        return (
            self.means_3d,
            self.scales,
            self.quats,
            self.rgbs,
            self.opacities,
        )

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass
    
    def obtain_data(self):
        return (
            self.means_3d[self.persistent_mask],
            self.scales[self.persistent_mask],
            self.quats[self.persistent_mask],
            torch.sigmoid(self.rgbs[self.persistent_mask]),
            torch.sigmoid(self.opacities[self.persistent_mask]),
        )

    def maskout(self, indices):
        self.means_3d.data[indices, :] = 0.
        self.scales.data[indices, :] = 0.
        self.quats.data[indices, :] = 0.
        self.rgbs.data[indices, :] = 0.
        self.opacities.data[indices, :] = 0.
        self.persistent_mask[indices] = False
    
    def maskout_grad(self):
        self.means_3d.grad[~self.persistent_mask, :] = 0.
        self.scales.grad[~self.persistent_mask, :] = 0.
        self.quats.grad[~self.persistent_mask, :] = 0.
        self.rgbs.grad[~self.persistent_mask, :] = 0.
        self.opacities.grad[~self.persistent_mask, :] = 0.
    
    @property
    def colors(self):
        rgbs = self.rgbs[self.persistent_mask]
        opacities = self.opacities[self.persistent_mask]

        clamped_rgbs = torch.sigmoid(rgbs)
        clamped_opacities = torch.sigmoid(opacities)

        return clamped_rgbs * clamped_opacities

    @property
    def current_num_samples(self):
        return float(self.persistent_mask.sum())

    def _init_gaussians(self, num_primarys, num_backups, device):
        num_points = num_primarys + num_backups

        self.means_3d = 2 * (torch.rand(num_points, 3, device=device) - 0.5)    # change to x y
        self.scales = torch.rand(num_points, 3, device=device) * 0.5

        u = torch.rand(num_points, 1, device=device)
        v = torch.rand(num_points, 1, device=device)
        w = torch.rand(num_points, 1, device=device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((num_points, 1), device=device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )
        
        self.background = torch.zeros(15, device=device)

        self.means_3d.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        # self.rgbs = torch.rand(num_points, 15, device=device)
        self.rgbs = torch.randn(num_points, 15, device=device) / 0.4
        self.rgbs.requires_grad = True

    def _init_mask(self, num_primarys, num_backups):
        self.persistent_mask = torch.cat(
            [torch.ones(num_primarys, dtype=bool), torch.zeros(num_backups, dtype=bool)], dim=0
        )

        self.current_marker = num_primarys
    
    def init_rgbs(self, xys, gt_images):
        # color range (color_bias, 1 - color_bias)
        color = obtain_init_color(
            input_xys=xys,
            hw=[self.H, self.W],
            image=gt_images,
        )

        # y = 1 / (1 + exp(-x)), exp(-x) = 1 / y - 1, x = - log(1 / y - 1)
        self.rgbs.data = - torch.log(1 / (color * (1 - 2e-8) + 1e-8) - 1)

    def post_colors(self, xys, gt_images, th=0.05):
        processed_colors = filter_by_background(
            xys=xys,
            colors=self.colors,
            hw=[self.H, self.W],
            image=gt_images,
            th=th,
        )

        return processed_colors
    
    def obtain_calibration(self, rna_class, rna_name):
        cos_score, pred_rna_index = obtain_simi(pred_code=self.colors, codebook=rna_class)
        pred_rna_name = rna_name[pred_rna_index.cpu().numpy()]

        return cos_score, pred_rna_index, pred_rna_name

    @torch.no_grad()
    def save_to_csv(self, xys, batch, rna_class, rna_name, hw, post_th, path):
        max_color_post = self.post_colors(xys, batch, th=post_th)  # (n, 15)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score, pred_rna_index, pred_rna_name = self.obtain_calibration(
            rna_class, 
            rna_name,
        )

        ref_score = cos_score * max_color_post

        write_to_csv(
            xys=xys,
            scores=torch.stack([ref_score, cos_score, max_color_post], dim=-1),
            hw=hw,
            rna_index=pred_rna_index,
            rna_name=pred_rna_name,
            path=path,
        )

    @torch.no_grad()
    def visualize_points(self, xys, batch, mdp_dapi_image, post_th, rna_class, rna_name):
        points_xy = xys.cpu().numpy()
        mdp_dapi_image = mdp_dapi_image.cpu().numpy()
        mdp_image = batch.max(dim=-1)[0].cpu().numpy()

        max_color = self.colors.max(dim=-1)[0]
        max_color = max_color / (max_color.max() + 1e-8)

        max_color_post = self.post_colors(xys, batch, th=post_th)  # (n, 15)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score = self.obtain_calibration(rna_class, rna_name)[0]

        ref_score = cos_score * max_color_post
        
        view_on_image = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color.cpu().numpy())
        view_on_image_post = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color_post.cpu().numpy())
        view_on_image_cos = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=cos_score.cpu().numpy())
        view_on_image_ref = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=ref_score.cpu().numpy())

        # view_on_dapi = view_positions(points_xy=points_xy, bg_image=mdp_dapi_image, alpha=max_color)
        # view_on_dapi_post = view_positions(points_xy=points_xy, bg_image=mdp_dapi_image, alpha=max_color_post)
        # view_on_dapi_cos = view_positions(points_xy=points_xy, bg_image=mdp_dapi_image, alpha=cos_score)
        # view_on_dapi_ref = view_positions(points_xy=points_xy, bg_image=mdp_dapi_image, alpha=ref_score)

        return view_on_image, view_on_image_post, view_on_image_cos, view_on_image_ref

class FixGaussModel(GaussModel):
    def obtain_data(self):
        return (
            self.means_3d,
            self.scales,
            self.quats,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
        )

    @property
    def colors(self):
        clamped_rgbs = torch.sigmoid(self.rgbs)
        clamped_opacities = torch.sigmoid(self.opacities)

        return clamped_rgbs * clamped_opacities

    @property
    def current_num_samples(self):
        return self.rgbs.shape[0]


model_zoo = {
    "gauss": GaussModel,
    "fix_gauss": FixGaussModel,
}