import math
import torch
import torch.nn as nn


class SliceCamera(nn.Module):
    def __init__(
        self, 
        num_slice,
        hw,
        num_dims,
        step_z=0.05, 
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

        self.viewmat = torch.tensor(
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

        self.slice_zs = (torch.linspace(0, num_slice, num_slice, device=device) + 0.5) / num_slice * step_z

    @property
    def z_planes(self):
        return self.slice_zs + self.camera_z