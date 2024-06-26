import os
import math
import torch
import numpy as np

from utils.utils import obtain_init_color, filter_by_background, write_to_csv
from utils.vis2d_utils import view_positions, view_score_dist
from utils.vis3d_utils import view_3d
from systems.losses import obtain_simi
from systems.render import render_single_slice, render_multi_slices, project_to_xys


class GaussModel(torch.nn.Module):
    def __init__(self, num_primarys, num_backups, device, camera, B_SIZE=16):
        super(GaussModel, self).__init__()
        self.camera = camera
        self._init_gaussians(num_primarys, num_backups, device)
        self._init_mask(num_primarys, num_backups)

        self.B_SIZE = B_SIZE
    
    def render_slice(self, camera, cam_idxs, slice_idxs):
        means_3d, scales, quats, rgbs, opacities = self.obtain_data()

        if len(slice_idxs) > 1:
            outputs_dict = {i: [] for i in range(4)}
            for cam_idx, slice_idx in zip(cam_idxs, slice_idxs):
                # (1, h, w, c), (n, 3), (n,), (n, 2), out_imgs, conics, radii, xys
                outputs = render_single_slice(
                    means_3d=means_3d, scales=scales, quats=quats, rgbs=rgbs, opacities=opacities, 
                    background=self.background, B_SIZE=self.B_SIZE,
                    viewmat=camera.viewmat(cam_idx), 
                    plane_zs=camera.plane_zs(cam_idx)[slice_idx],
                    focal=camera.focal, hw=camera.hw, 
                )
                for i, x in enumerate(outputs):
                    outputs_dict[i].append(x) 

            return [torch.cat(outputs_dict[i], dim=0) for i in range(4)]
            
        elif len(slice_idxs) == 1:
            return render_single_slice(
                means_3d=means_3d, scales=scales, quats=quats, rgbs=rgbs, opacities=opacities, 
                background=self.background, B_SIZE=self.B_SIZE,
                viewmat=camera.viewmat(cam_idxs), 
                plane_zs=camera.plane_zs(cam_idxs)[slice_idxs],
                focal=camera.focal, hw=camera.hw, 
            )
        else:
            raise ValueError("Plane index should be at least one element.")
    
    def obtain_xys(self, camera):
        return project_to_xys(
            means_3d=self.obtain_data()[0], 
            viewmat=camera.viewmat(0), 
            img_size=(camera.hw[1], camera.hw[0]),
        )
        
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
            torch.nn.functional.softplus(self.scales[self.persistent_mask]),
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

        self.means_3d = 2 * (torch.rand(num_points, 3, device=device) - 0.5)    # [-1, 1]

        # radii ~ s * 3 * (w / 2) / 8 = s * w * 3 / 16
        # point size (2-5 px), set range (0px, 12px), 128-0.5, 256-0.25, 64 / 128 = 0.5, 64 / 64
        # self.scales = torch.rand(num_points, 3, device=device)
        self.scales = torch.zeros(num_points, 3, device=device) * 0.5

        # u = torch.rand(num_points, 1, device=device)
        # v = torch.rand(num_points, 1, device=device)
        # w = torch.rand(num_points, 1, device=device)

        # self.quats = torch.cat(
        #     [
        #         torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        #         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        #         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        #         torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        #     ],
        #     -1,
        # )

        self.quats = torch.zeros(num_points, 4, device=device)
        self.quats[:, 0] = 1.
        
        self.opacities = torch.ones((num_points, 1), device=device)
        self.background = torch.zeros(self.camera.num_dims, device=device)

        self.means_3d.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.opacities.requires_grad = True

        self.rgbs = torch.randn(num_points, self.camera.num_dims, device=device)
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
            hw=self.camera.hw,
            image=gt_images,
        )

        # y = 1 / (1 + exp(-x)), exp(-x) = 1 / y - 1, x = - log(1 / y - 1)
        self.rgbs.data = - torch.log(1 / (color * (1 - 2e-8) + 1e-8) - 1)

    def post_colors(self, xys, gt_images, th=0.05):
        processed_colors = filter_by_background(
            xys=xys,
            colors=self.colors,
            hw=self.camera.hw,
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
    def visualize_3d(self, save_path):
        means_3d, scales, quats = self.obtain_data()[:3]
        colors = self.colors
        
        mask = (torch.abs(means_3d[:, 0]) <= 1) & (torch.abs(means_3d[:, 1]) <= 1) & (torch.min(colors, dim=-1)[0] > 0.2) & (torch.min(scales, dim=-1)[0] > 0.01)

        # sort by color and only draw top 50000 points
        mask = torch.argsort(colors, descending=True)[0][:50000]
         
        means_3d = means_3d[mask].data.cpu().numpy()
        scales = scales[mask].data.cpu().numpy()
        quats = quats[mask].data.cpu().numpy()
        # colors = colors[mask].data.cpu().numpy()

        view_3d(
            means_3d=means_3d, 
            scales=scales,
            quats=quats, 
            colors=None, 
            save_path=save_path,
        )


class FixGaussModel(GaussModel):
    def obtain_data(self):
        return (
            self.means_3d,
            torch.nn.functional.softplus(self.scales),
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