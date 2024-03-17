import math
import torch

from datasets import obtain_init_color

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
    
    def obtain_data(self, sacle=1.0):
        return (
            self.means_3d[self.persistent_mask],
            self.scales[self.persistent_mask],
            self.quats[self.persistent_mask],
            torch.clamp(torch.sigmoid(self.rgbs[self.persistent_mask]) * sacle + 0.5 * (1 - sacle), 0, 1),
            torch.clamp(torch.sigmoid(self.opacities[self.persistent_mask]) * sacle + 0.5 * (1 - sacle), 0, 1),
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
    
    def colors(self, scale=1.0):
        rgbs = self.rgbs[self.persistent_mask]
        opacities = self.opacities[self.persistent_mask]

        clamped_rgbs = torch.clamp(torch.sigmoid(rgbs) * scale + 0.5 * (1 - scale), 0, 1)
        clamped_opacities = torch.clamp(torch.sigmoid(opacities) * scale + 0.5 * (1 - scale), 0, 1)

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
        color = obtain_init_color(
            input_xys=xys,
            hw=[self.H, self.W],
            image=gt_images,
        )

        # y = 1.1 / (1 + exp(-x)), exp(-x) = 1.1 / y - 1, x = - log(1.1 / y - 1)
        self.rgbs.data = - torch.log(1 / color - 1)


class FixGaussModel(GaussModel):
    def obtain_data(self, scale=1.0):
        return (
            self.means_3d,
            self.scales,
            self.quats,
            torch.clamp(torch.sigmoid(self.rgbs) * scale + 0.5 * (1 - scale), 0, 1),
            torch.clamp(torch.sigmoid(self.opacities) * scale + 0.5 * (1 - scale), 0, 1),
        )

    def colors(self, scale=1.0):
        clamped_rgbs = torch.clamp(torch.sigmoid(self.rgbs) * scale + 0.5 * (1 - scale), 0, 1)
        clamped_opacities = torch.clamp(torch.sigmoid(self.opacities) * scale + 0.5 * (1 - scale), 0, 1)

        return clamped_rgbs * clamped_opacities

    @property
    def current_num_samples(self):
        return self.rgbs.shape[0]


model_zoo = {
    "gauss": GaussModel,
    "fix_gauss": FixGaussModel,
}