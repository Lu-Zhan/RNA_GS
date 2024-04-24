import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from utils.gaussian_utils import slice_3d_gaussians_by_single_z, slice_3d_gaussians_by_multi_z


def render_single_slice(
        means_3d, scales, quats, rgbs, opacities, background, 
        camera, B_SIZE, index,
    ):

    cov3d, cov2d, xys, depths, radii, conics, num_tiles_hit, mask, scale_term = slice_3d_gaussians_by_single_z(
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


def render_multi_slices(
        means_3d, scales, quats, rgbs, opacities, background, 
        camera, B_SIZE, index,
    ):

    # index: (k,)

    # 
    cov3d, cov2d, xys, depths, radii, conics, num_tiles_hit, mask, scale_term = slice_3d_gaussians_by_multi_z(
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

    new_rgbs = rgbs[..., None] * scale_term[:, None, :] # (n, c, 1) * (n, 1, k) -> (n, c, k)

    out_imgs = []
    for k in range(new_rgbs.shape[-1]):
        out_img = rasterize_gaussians(
            xys=xys[..., k], 
            depths=depths[..., k],
            radii=radii[..., k],
            conics=conics[..., k],
            num_tiles_hit=num_tiles_hit[..., k],
            colors=new_rgbs[..., k],
            opacity=opacities,
            img_height=camera.H, img_width=camera.W, block_width=B_SIZE, background=background,
        )

        out_imgs.append(out_img)
    
    out_imgs = torch.stack(out_imgs, dim=0)

    # return out_imgs, conics, radii, xys
    return out_imgs, conics, radii, xys



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