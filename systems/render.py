import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians


def render_slices_2d(
        means_3d, scales, quats, rgbs, opacities, background, 
        camera, B_SIZE, index,
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

    xys_z, conics_z, new_rgbs = _obtain_slice_info(
        means_3d=means_3d, cov3d=cov3d, rgbs=rgbs, xys=xys,
        camera=camera, index=index,
    )
    
    out_imgs = []
    for k in range(xys_z.shape[-1]):
        out_img = rasterize_gaussians(
            xys=xys_z[..., k], 
            depths=depths,
            radii=radii,
            conics=conics_z[..., k],
            num_tiles_hit=num_tiles_hit,
            colors=new_rgbs[..., k],
            opacity=opacities,
            img_height=camera.H, img_width=camera.W, block_width=B_SIZE, background=background,
        )

        out_imgs.append(out_img)
    
    out_imgs = torch.stack(out_imgs, dim=0) # (k, h, w, 15)
    # torch.cuda.empty_cache()

    return out_imgs, conics, radii, xys


def _obtain_slice_info(
        means_3d, cov3d, rgbs, xys,
        camera, index,
    ):
    # cov3d: (n, 6), [
    #     0, 1, 2,
    #     x, 3, 4,
    #     x, x, 5,
    # ]

    U = torch.stack([cov3d[..., 0], cov3d[..., 1], cov3d[..., 3]], dim=-1)  # (n, 3), up triangle of matrix
    V = torch.stack([cov3d[..., 2], cov3d[..., 4]], dim=-1) # (n, 2)
    W = cov3d[..., -1:] # (n, 1)

    if index is None:
        index = torch.arange(camera.plane_zs.shape[0], device=means_3d.device, dtype=torch.long)

    delta_z = camera.plane_zs[None, index] - means_3d[:, -1:]   # (n, k)

    # x y z axis [-1, 1] -> [-(w / 2) / s, (w / 2) / s] -> * 8
    # transform conics and scale_term from 2D slice space to camera space        
    distance_z = camera.plane_zs[None, index] - camera.camera_z.reshape(1, 1)   # (n, k)
    s = camera.focal / (distance_z + 1e-8)   # 64 / 8 = 8 
    delta_z_px = delta_z * s   # (n, k)

    lam = 1 / (W + 1e-8)    # (n, 1)
    scale_term = torch.exp(-0.5 * lam * delta_z_px ** 2)   # (n, k)
    rgbs_z = rgbs[..., None] * scale_term[:, None, :]

    # mean xy from the slice of cov3d given z
    # (n, 2, 1) + (n, 2, 1) / (n, 1, 1) * (n, 1, k) -> (n, 2, k)
    xys_z = xys[..., None] + V[..., None] / (W[..., None] + 1e-8) * delta_z[:, None, :] * camera.W / 2  # (n, 2, k)
    
    # cov2d from the slice of cov3d given z
    VV_t = torch.stack([V[..., 0] ** 2, V[..., 0] * V[..., 1], V[..., 1] ** 2], dim=-1)  # (n, 3)
    cov2d_z = U - VV_t / (W + 1e-8)
    
    # conics_z is the inverse version of cov2d_z
    det_cov2d_z = cov2d_z[..., 0] * cov2d_z[..., 2] - cov2d_z[..., 1] ** 2 + 1e-8
    # c, -b, a / det
    conics_z = torch.stack([cov2d_z[..., 2] / det_cov2d_z, -cov2d_z[..., 1] / det_cov2d_z, cov2d_z[..., 0] / det_cov2d_z], dim=-1)  # (n, 3)

    # rescale the conics_z by considering the z scaling
    # (n, 3, 1) / (n, 1, k) -> (n, 3, k)
    conics_z = conics_z[..., None] / (s[:, None, :] ** 2 + 1e-8) 

    return xys_z, conics_z, rgbs_z


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