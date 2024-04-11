import torch
import numpy as np
import matplotlib.pyplot as plt


from systems.models import FixGaussModel
from systems.cameras import SliceCamera
from gsplat import project_gaussians, rasterize_gaussians

class SliceFixGaussModel(FixGaussModel):
    def render_slice(self, z_plane):
        means_3d, scales, quats, rgbs, opacities = self.obtain_data()

        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
            means3d=means_3d,
            scales=scales,
            glob_scale=1,
            quats=quats,
            viewmat=self.viewmat,
            projmat=self.viewmat,
            fx=self.focal, fy=self.focal, cx=self.W / 2, cy=self.H / 2,
            img_height=self.H, img_width=self.W, block_width=self.B_SIZE,
        )

        # inv_c33 = cov3d[..., -1]
        # c33 = 1 / inv_c33
        # camera_z = - self.viewmat[2, 3]

        # term = torch.exp(-0.5 * c33 * (z_plane - means_3d[..., -1]) ** 2)
        # rescale_radii = (means_3d[..., -1] - camera_z) /  (z_plane - camera_z) * term

        # new_radii = radii # * rescale_radii
        # new_radii = new_radii.to(radii.dtype)

        # depths = torch.ones_like(depths) * z_plane - camera_z

        new_conics = conics / z_plane

        out_img = rasterize_gaussians(
            xys=xys, 
            depths=depths,
            radii=radii,
            conics=new_conics,
            num_tiles_hit=num_tiles_hit,
            colors=rgbs,
            opacity=opacities,
            img_height=self.H, img_width=self.W, block_width=self.B_SIZE, background=self.background,
        )

        return out_img, new_conics, radii, xys


gs_model = SliceFixGaussModel(num_primarys=1, num_backups=0, hw=(128, 128), device='cuda:0')
gs_model.rgbs = torch.ones_like(gs_model.rgbs)[..., :1]
gs_model.opacities = torch.ones_like(gs_model.opacities)[..., :1]
gs_model.background = torch.zeros_like(gs_model.background)[..., :1]
gs_model.means_3d = torch.zeros_like(gs_model.means_3d)
gs_model.means_3d[..., -1] = 1
gs_model.scales = torch.ones_like(gs_model.scales)

z_plane = 1
image, _, radii, _ = gs_model.render_slice(z_plane=z_plane)
plt.subplot(1, 2, 1)
plt.imshow(image.data.cpu().numpy(), cmap='gray')
plt.title(f'z={z_plane:.1f}')

z_plane = 2
image, _, radii, _ = gs_model.render_slice(z_plane=z_plane)
plt.subplot(1, 2, 1)
plt.imshow(image.data.cpu().numpy(), cmap='gray')
plt.title(f'z={z_plane:.1f}')


plt.tight_layout()
plt.axis('off')
plt.savefig('slice test.png')

# w = 5
# h = 5
# bias = 2
# # image, _, radii, _ = gs_model.render()

# # plt.subplot(h, w, 1)
# # plt.imshow(image.data.cpu().numpy(), cmap='gray')
# # plt.title(f'z=0')

# for i in range(0, h * w):
#     z_plane = 2 * bias / (h * w) * i - bias
#     image, _, radii, _ = gs_model.render_slice(z_plane=z_plane)
#     plt.subplot(h, w, i+1)
#     plt.imshow(image.data.cpu().numpy(), cmap='gray')
#     plt.title(f'z={z_plane:.1f}')
#     plt.tight_layout()
#     plt.axis('off')

# plt.savefig('slice test.png')
