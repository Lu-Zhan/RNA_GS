import math
from typing import Iterator
import torch
import torch.nn as nn
    

class SliceCameras(nn.Module):
    def __init__(
        self, 
        max_num_cams,
        max_num_slices,
        num_dims,
        hw,
        step_z=0.025, 
        camera_z=-8,
        refine_camera=False,
        device=torch.device('cuda:0'),
        ):
        super(SliceCameras, self).__init__()

        fov_x = math.pi / 2.0
        self.hw = hw
        self.focal = 0.5 * float(hw[1]) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([hw[1], hw[0], 1], device=device)

        self.device = device
        self.num_dims = num_dims

        self._init_camera(max_num_cams, max_num_slices, camera_z, step_z)

    def _init_camera(self, num_cams, num_slices, cam_z, step_z):
        self.base_camera_zs = torch.ones(num_cams, device=self.device).reshape(-1) * cam_z
        self.camera_shift = torch.zeros(num_cams - 1, device=self.device)
        self.camera_shift.requires_grad = True

        # self.current_viewmat = torch.eye(4, device=self.device).unsqueeze(0).repeat(num_cams, 1, 1)   # (num_cams, 4, 4)
        # self.current_viewmat[:, 2, -1] = -self.camera_zs

        self.current_plane_zs = [-1 + torch.arange(num_slices, device=self.device) * step_z - cam_z for _ in range(num_cams)]

    def viewmat(self, cam_index):
        view_mat = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -self.camera_zs[cam_index]],
            [0, 0, 0, 1],
        ], device=self.camera_zs.device, dtype=self.camera_zs.dtype)

        return view_mat
    
    def plane_zs(self, cam_index):
        return self.current_plane_zs[cam_index] + self.camera_zs[cam_index]
    
    @property
    def camera_zs(self):
        all_camera_shift = torch.cat([torch.zeros_like(self.camera_shift[:1]), self.camera_shift], dim=0)
        return self.base_camera_zs + all_camera_shift
    
    @property
    def parameters(self):
        return [self.camera_shift]
    
    def load_camera_shift(self, camera_shift):
        self.camera_shift.data = camera_shift