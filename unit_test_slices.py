import torch

from matplotlib import pyplot as plt
from systems.cameras import SliceCamera
from systems.models import FixGaussModel
from gsplat import project_gaussians, rasterize_gaussians

class SliceFixGaussModel(FixGaussModel):    
    def render_z_plane(self, z_plane):
        means_3d, scales, quats, rgbs, opacities = self.obtain_data()

        camera = self.camera

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

        # inv_c33 = cov3d[..., -1]
        # c33 = 1 / inv_c33
        # camera_z = - self.viewmat[2, 3]

        # term = torch.exp(-0.5 * c33 * (z_plane - means_3d[..., -1]) ** 2)
        # rescale_radii = (means_3d[..., -1] - camera_z) /  (z_plane - camera_z) * term

        # new_radii = radii / z_plane ** 4 # * rescale_radii
        # new_radii = new_radii.to(radii.dtype)

        # depths = torch.ones_like(depths) * z_plane - camera_z

        new_conics = conics * z_plane / 

        # opacities = 

        out_img = rasterize_gaussians(
            xys=xys, 
            depths=depths,
            radii=radii,
            conics=new_conics,
            num_tiles_hit=num_tiles_hit,
            colors=rgbs,
            opacity=opacities,
            img_height=camera.H, img_width=camera.W, block_width=self.B_SIZE, background=self.background,
        )

        return out_img, new_conics, radii, xys
    

if __name__ == '__main__':
    camera = SliceCamera(
        num_slice=1,
        num_dims=1,
        hw=(128, 128),
        step_z=0.01,
        camera_z=-8,
    )

    gs_model = SliceFixGaussModel(num_primarys=1, num_backups=0, camera=camera, device='cuda:0')
    gs_model.rgbs = torch.zeros_like(gs_model.rgbs)[..., :1]
    gs_model.opacities = torch.ones_like(gs_model.opacities)[..., :1]
    gs_model.background = torch.zeros_like(gs_model.background)[..., :1]
    gs_model.means_3d = torch.zeros_like(gs_model.means_3d)
    gs_model.means_3d[..., -1] = 1
    gs_model.scales = torch.ones_like(gs_model.scales)

    z_plane = 1
    image, _, radii, _ = gs_model.render_z_plane(z_plane=z_plane)
    plt.subplot(1, 2, 1)
    plt.imshow(image.data.cpu().numpy(), cmap='gray', vmax=0.5)
    plt.title(f'z={z_plane:.1f}')

    z_plane = 2
    image, _, radii, _ = gs_model.render_z_plane(z_plane=z_plane)
    plt.subplot(1, 2, 2)
    plt.imshow(image.data.cpu().numpy(), cmap='gray', vmax=0.5)
    plt.title(f'z={z_plane:.1f}')

    # plt.tight_layout()
    # plt.axis('off')
    plt.savefig('new.png')
    