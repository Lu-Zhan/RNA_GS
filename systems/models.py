import os
import math
import torch
import numpy as np

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from utils.utils import obtain_init_color, filter_by_background, write_to_csv
from utils.visualize import view_positions, view_score_dist
from systems.losses import obtain_simi
from plyfile import PlyData, PlyElement
from os import makedirs, path
from errno import EEXIST

class GaussModel(torch.nn.Module):
    def __init__(self, num_primarys, num_backups, device, camera, B_SIZE=16):
        super(GaussModel, self).__init__()
        self.camera = camera
        self._init_gaussians(num_primarys, num_backups, device)
        self._init_mask(num_primarys, num_backups)

        self.B_SIZE = B_SIZE
    
    def render(self, camera=None):
        if camera is None:
            camera = self.camera

        means_3d, scales, quats, rgbs, opacities = self.obtain_data()

        # legacy code
        try:
            self.B_SIZE += 0
        except:
            self.B_SIZE = 16

        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
            means3d=means_3d,
            scales=scales,
            glob_scale=1,
            quats=quats,
            viewmat=camera.viewmat,
            projmat=camera.viewmat,
            fx=camera.focal, fy=camera.focal, cx=camera.W / 2, cy=camera.H / 2,
            img_height=camera.H, img_width=camera.W, block_width=self.B_SIZE,
        )

        out_img = rasterize_gaussians(
            xys=xys, 
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=rgbs,
            opacity=opacities,
            img_height=camera.H, img_width=camera.W, block_width=self.B_SIZE, background=self.background,
        )

        return out_img, conics, radii, xys
    
    def render_slices(self, camera=None):
        if camera is None:
            camera = self.camera

        means_3d, scales, quats, rgbs, opacities = self.obtain_data()

        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
            means3d=means_3d,
            scales=scales,
            glob_scale=1,
            quats=quats,
            viewmat=camera.viewmat,
            projmat=camera.viewmat,
            fx=camera.focal, fy=camera.focal, cx=camera.W / 2, cy=camera.H / 2,
            img_height=camera.H, img_width=camera.W, block_width=self.B_SIZE,
        )

        c33 = 1 / cov3d[..., -1:]
        camera_z = camera.camera_z
        plane_zs = camera.plane_zs

        # means_3d (n, 3), z_plane (k)

        # (1, k) - (n, 1) = (n, k), (n, 1) - (n, k) = (n, k)
        term = torch.exp(-0.5 * c33 * (plane_zs[None, :] - means_3d[:, -1:]) ** 2)

        # (n, 1) / (1, k) = (n, k) * (n, k) = (n, k)
        rescale_radii = (means_3d[..., -1:] - camera_z) /  (plane_zs[None, :] - camera_z) * term

        # (n, 1, c) / (n, k, 1) = (n, k, c), add 1e-8 to avoid zero division
        new_conics = conics[:, None, :] / (rescale_radii[..., None] ** 2 + 1e-8)

        out_imgs = []
        for k in range(new_conics.shape[1]):
            out_img = rasterize_gaussians(
                xys=xys, 
                depths=depths,
                radii=radii,
                conics=new_conics[:, k],
                num_tiles_hit=num_tiles_hit,
                colors=rgbs,
                opacity=opacities,
                img_height=camera.H, img_width=camera.W, block_width=self.B_SIZE, background=self.background,
            )

            out_imgs.append(out_img)
        
        out_imgs = torch.stack(out_imgs, dim=0) # (k, h, w, 15)
        return out_imgs, new_conics, radii, xys
    
    @property
    def parameters(self):
        return (
            self.means_3d,
            self.scales,
            self.quats,
            self.rgbs,
            self.opacities,
        )
    
    def obtain_data(self):
        return (
            self.means_3d[self.persistent_mask],
            self.scales[self.persistent_mask],
            self.quats[self.persistent_mask],
            torch.sigmoid(self.rgbs[self.persistent_mask]),
            torch.sigmoid(self.opacities[self.persistent_mask]),
        )
    
    def construct_list_of_attributes(self):
        means_3d, scales, quats, rgbs, opacities = self.obtain_data()
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(3):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(2):
            l.append('rot_{}'.format(i))
        return l


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

        # self.viewmat = torch.tensor(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 8.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device=device,
        # )
        
        self.background = torch.zeros(self.camera.num_dims, device=device)

        self.means_3d.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.opacities.requires_grad = True
        # self.viewmat.requires_grad = False

        # self.rgbs = torch.rand(num_points, 15, device=device)
        self.rgbs = torch.randn(num_points, self.camera.num_dims, device=device) / 0.4
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
            hw=[self.camera.H, self.camera.W],
            image=gt_images,
        )

        # y = 1 / (1 + exp(-x)), exp(-x) = 1 / y - 1, x = - log(1 / y - 1)
        self.rgbs.data = - torch.log(1 / (color * (1 - 2e-8) + 1e-8) - 1)

    def post_colors(self, xys, gt_images, th=0.05):
        processed_colors = filter_by_background(
            xys=xys,
            colors=self.colors,
            hw=[self.camera.H, self.camera.W],
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

        # filter out points with zero score
        mask = ref_score > 0

        xys = xys[mask]
        ref_score = ref_score[mask]
        cos_score = cos_score[mask]
        max_color_post = max_color_post[mask]
        pred_rna_index = pred_rna_index[mask]
        pred_rna_name = pred_rna_name[mask.cpu().numpy()]

        write_to_csv(
            xys=xys,
            scores=torch.stack([ref_score, cos_score, max_color_post], dim=-1),
            hw=hw,
            rna_index=pred_rna_index,
            rna_name=pred_rna_name,
            path=path,
        )

    @torch.no_grad()
    def visualize_points(
            self, xys, batch, mdp_dapi_image, post_th, rna_class, rna_name, 
            selected_classes=['Snap25', 'Slc17a7', 'Gad1', 'Gad2', 'Plp1', 'Mbp', 'Aqp4', 'Rgs5']
        ):
        points_xy = xys.cpu().numpy()
        mdp_dapi_image = mdp_dapi_image.cpu().numpy()
        mdp_image = batch.max(dim=-1)[0].cpu().numpy()

        max_color = self.colors.max(dim=-1)[0]
        max_color = max_color / (max_color.max() + 1e-8)

        max_color_post = self.post_colors(xys, batch, th=post_th)  # (n, 15)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score, _, pred_class_name = self.obtain_calibration(rna_class, rna_name)

        ref_score = cos_score * max_color_post

        # filter out points with zero score
        mask = ref_score > post_th
        pred_class_name = pred_class_name[mask.cpu().numpy()]
        points_xy = points_xy[mask.cpu().numpy()]
        ref_score = ref_score[mask]
        cos_score = cos_score[mask]
        max_color_post = max_color_post[mask]
        max_color = max_color[mask]
        
        view_on_image = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color.cpu().numpy())
        view_on_image_post = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color_post.cpu().numpy())
        view_on_image_cos = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=cos_score.cpu().numpy())
        view_on_image_ref = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=ref_score.cpu().numpy())

        # show selected rna classes
        view_classes = []
        for selected_class in selected_classes:
            selected_index = np.where(pred_class_name == selected_class)[0]
            
            # if len(selected_index) > 0:
            selected_points = points_xy[selected_index]
            selected_ref_score = ref_score[selected_index]

            # hanming weight for selected class
            hm_weight = int(rna_class[np.where(rna_name == selected_class)[0]].sum())

            view_specific = view_positions(
                points_xy=selected_points, bg_image=mdp_dapi_image, alpha=selected_ref_score.cpu().numpy(), 
                s=3, prefix=f"{selected_class}-{hm_weight:01d}: "
            )

            view_classes.append(view_specific)

        return view_on_image, view_on_image_post, view_on_image_cos, view_on_image_ref, view_classes
    
    @torch.no_grad()
    def visualize_top_classes(
            self, xys, batch, mdp_dapi_image, post_th, rna_class, rna_name, top_k=10,
        ):
        points_xy = xys.cpu().numpy()
        mdp_dapi_image = mdp_dapi_image.cpu().numpy()

        max_color = self.colors.max(dim=-1)[0]
        max_color = max_color / (max_color.max() + 1e-8)

        max_color_post = self.post_colors(xys, batch, th=post_th)  # (n, 15)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score, _, pred_class_name = self.obtain_calibration(rna_class, rna_name)

        ref_score = cos_score * max_color_post

        # filter out points with zero score
        mask = ref_score > 0
        pred_class_name = pred_class_name[mask.cpu().numpy()]
        points_xy = points_xy[mask.cpu().numpy()]
        ref_score = ref_score[mask]

        # count the sum number of each class and find top k classes
        class_count = {}
        for class_name in pred_class_name:
            if class_name in class_count:
                class_count[class_name] += 1
            else:
                class_count[class_name] = 1
        
        sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
        selected_classes = [class_name for class_name, _ in sorted_class_count[:top_k]]

        # show selected rna classes
        view_classes = []
        for selected_class in selected_classes:
            selected_index = np.where(pred_class_name == selected_class)[0]
            
            # if len(selected_index) > 0:
            selected_points = points_xy[selected_index]
            selected_ref_score = ref_score[selected_index]

            # hanming weight for selected class
            hm_weight = int(rna_class[np.where(rna_name == selected_class)[0]].sum())

            view_specific = view_positions(
                points_xy=selected_points, bg_image=mdp_dapi_image, alpha=selected_ref_score.cpu().numpy(), 
                s=3, prefix=f"{selected_class}-{hm_weight:01d}: "
            )

            view_classes.append(view_specific)

        return view_classes, selected_classes

    @torch.no_grad()
    def visualize_score_dist(
            self, xys, batch, post_th, rna_class, rna_name, save_folder,
            selected_classes=['Snap25', 'Slc17a7', 'Gad1', 'Gad2', 'Plp1', 'Mbp', 'Aqp4', 'Rgs5']
        ):
        max_color_post = self.post_colors(xys, batch, th=post_th)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score, _, pred_class_name = self.obtain_calibration(rna_class, rna_name)

        ref_score = cos_score * max_color_post

        # selected_classes=self.hparams['view']['classes']
        view_score_dist(selected_classes, pred_class_name, ref_score, rna_class, rna_name, save_folder)
    
    @torch.no_grad()
    def save_3d_pts(self, xys, batch, rna_class, rna_name, post_th, path):
        
        max_color_post = self.post_colors(xys, batch, th=post_th)  # (n, 15)
        max_color_post = max_color_post.max(dim=-1)[0]
        max_color_post = max_color_post / (max_color_post.max() + 1e-8)

        cos_score, pred_rna_index, pred_rna_name = self.obtain_calibration(
            rna_class, 
            rna_name,
        )

        ref_score = cos_score * max_color_post

        # filter out points with zero score
        mask = ref_score > 0
        means_3d, _, _, _, _ = self.obtain_data()
        xys = xys[mask]
        z = means_3d[:, 2][mask]
        ref_score = ref_score[mask]
        cos_score = cos_score[mask]
        max_color_post = max_color_post[mask]
        pred_rna_index = pred_rna_index[mask]
        pred_rna_name = pred_rna_name[mask.cpu().numpy()]


        torch.save({
            'xy': torch.tensor(xys),
            'z': torch.tensor(z),
            'ref_score': torch.tensor(ref_score),
        }, path)


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