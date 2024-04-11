import math
import torch
import torch.nn as nn


class SliceCamera(nn.Module):
    def __init__(
        self, 
        num_slice,
        hw,
        num_dims,
        step_z=0.025, 
        refine_camera=False, 
        camera_z=-8,
        device=torch.device('cuda:0'),
        ):
        super(SliceCamera, self).__init__()

        fov_x = math.pi / 2.0
        self.H, self.W = hw[0], hw[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=device)

        self.device = device
        self.num_dims = num_dims

        self.base_viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -camera_z],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        self.camera_z = torch.tensor(camera_z, device=device)
        if refine_camera:
            self.camera_z.requires_grad = True

        self.base_plane_zs = (torch.arange(num_slice, device=device)) * step_z - num_slice / 2 * step_z - camera_z
    
    @property
    def viewmat(self):
        self.base_viewmat[2, -1] = -self.camera_z
        return self.base_viewmat
    
    @property
    def plane_zs(self):
        return self.base_plane_zs + self.camera_z