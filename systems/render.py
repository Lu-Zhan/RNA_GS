import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from utils.gaussian_utils import slice_3d_gaussians


def render_slice_2d(
        means_3d, scales, quats, rgbs, opacities, background, 
        camera, B_SIZE, index,
    ):

    cov3d, cov2d, xys, depths, radii, conics, num_tiles_hit, mask, scale_term = slice_3d_gaussians(
        means3d=means_3d,
        scales=scales,
        glob_scale=1,
        quats=quats,
        viewmat=camera.viewmat,
        fullmat=camera.viewmat,
        intrins=(camera.focal, camera.focal, camera.W / 2, camera.H / 2),
        plane_z=camera.plane_zs[index],
        img_size=(camera.W, camera.H),
        block_width=B_SIZE,
    )

    new_rgbs = rgbs * scale_term[:, None] # (n, c) * (n, 1) -> (n, c)

    out_img = rasterize_gaussians(
        xys=xys, 
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=new_rgbs,
        opacity=opacities,
        img_height=camera.H, img_width=camera.W, block_width=B_SIZE, background=background,
    )

    # return out_imgs, conics, radii, xys
    return out_img[None, ...], conics, radii, xys


def render_image(
        means_3d, scales, quats, rgbs, opacities, background,
        camera, B_SIZE
    ):

    xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
        means3d=means_3d,
        scales=scales,
        glob_scale=1,
        quats=quats,
        viewmat=camera.viewmat,
        projmat=camera.viewmat,
        fx=camera.focal, fy=camera.focal, cx=camera.W / 2, cy=camera.H / 2,
        img_height=camera.H, img_width=camera.W, block_width=B_SIZE,
    )

    out_img = rasterize_gaussians(
        xys=xys, 
        depths=depths,
        radii=radii,
        conics=conics,
        num_tiles_hit=num_tiles_hit,
        colors=rgbs,
        opacity=opacities,
        img_height=camera.H, img_width=camera.W, block_width=B_SIZE, background=background,
    )

    return out_img, conics, radii, xys