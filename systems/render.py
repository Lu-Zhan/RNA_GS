import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from utils.gaussian_utils import slice_3d_gaussians_by_single_z, slice_3d_gaussians_by_multi_z, project_pix


def render_single_slice(
        means_3d, scales, quats, rgbs, opacities, background, B_SIZE,
        viewmat, focal, plane_zs, hw,
    ):

    cov3d, cov2d, xys, depths, radii, conics, num_tiles_hit, mask, scale_term = slice_3d_gaussians_by_single_z(
        means3d=means_3d,
        scales=scales,
        glob_scale=1,
        quats=quats,
        viewmat=viewmat,
        fullmat=viewmat,
        intrins=(focal, focal, hw[1] / 2, hw[0] / 2),
        plane_z=plane_zs,
        img_size=(hw[0], hw[1]),
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
        img_height=hw[0], img_width=hw[1], block_width=B_SIZE, background=background,
    )

    # (1, h, w, c), (n, 3), (n,), (n, 2)
    return out_img[None, ...], cov2d, radii, xys


def render_multi_slices(
        means_3d, scales, quats, rgbs, opacities, background, B_SIZE,
        viewmat, focal, plane_zs, hw,
    ):

    # index: (k,)
    # 
    cov3d, cov2d, xys, depths, radii, conics, num_tiles_hit, mask, scale_term = slice_3d_gaussians_by_multi_z(
        means3d=means_3d,
        scales=scales,
        glob_scale=1,
        quats=quats,
        viewmat=viewmat,
        fullmat=viewmat,
        intrins=(focal, focal, hw[1] / 2, hw[0] / 2),
        plane_z=plane_zs,
        img_size=(hw[0], hw[1]),
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
            img_height=hw[0], img_width=hw[1], block_width=B_SIZE, background=background,
        )

        out_imgs.append(out_img)
    
    out_imgs = torch.stack(out_imgs, dim=0)

    # (k, h, w, c), (n, 3, k), (n, k), (n, 2, k)
    return out_imgs, cov2d, radii, xys


def project_to_xys(means_3d, viewmat, img_size):
    # xys = project_pix(fullmat, means3d, img_size, (cx, cy)) # (n * k, 2)
    return project_pix(
        fullmat=viewmat, 
        p=means_3d, 
        img_size=img_size, 
        center=(img_size[0] / 2, img_size[1] / 2),
    )
