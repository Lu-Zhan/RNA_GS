import torch
from gsplat._torch_impl import (
    get_tile_bbox, 
    project_pix, 
    compute_cov2d_bounds, 
    clip_near_plane, 
    scale_rot_to_cov3d, 
)

def slice_3d_gaussians(
    means3d,
    scales,
    glob_scale,
    quats,
    viewmat,
    fullmat,
    intrins,
    plane_z,
    img_size,
    block_width,
    clip_thresh=0.01,
):
    tile_bounds = (
        (img_size[0] + block_width - 1) // block_width,
        (img_size[1] + block_width - 1) // block_width,
        1,
    )
    fx, fy, cx, cy = intrins
    # tan_fovx = 0.5 * img_size[0] / fx
    # tan_fovy = 0.5 * img_size[1] / fy
    p_view, is_close = clip_near_plane(means3d, viewmat, clip_thresh)
    cov3d = scale_rot_to_cov3d(scales, glob_scale, quats)

    mean_shift, cov2d, scale_term = obtain_cov2d_and_mean_shift(
        means3d, cov3d, plane_z, torch.abs(viewmat[2, 3])
    )

    # shift xy at plane given z, depth is not change!
    means3d = means3d + mean_shift

    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    xys = project_pix(fullmat, means3d, img_size, (cx, cy))

    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    # compensation = torch.where(~mask, 0, compensation)
    num_tiles_hit = torch.where(~mask, 0, num_tiles_hit)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    i, j = torch.triu_indices(2, 2)
    cov2d_triu = cov2d[..., i, j]

    return (
        cov3d_triu,
        cov2d_triu,
        xys,
        depths,
        radii,
        conic,
        # compensation,
        num_tiles_hit,
        mask,
        scale_term,
    )


def obtain_cov2d_and_mean_shift(
        means_3d, cov3d, plane_z, ratio_x_z,
    ):

    U = cov3d[:, :2, :2]    # (n, 2, 2)
    V = cov3d[:, :2, 2] # (n, 2)
    W = cov3d[:, 2, 2] # (n)

    delta_z = plane_z - means_3d[:, -1]   # (n)

    delta_z_px = delta_z * ratio_x_z   # (n)

    lam = 1 / (W + 1e-8)    # (n)
    scale_term = torch.exp(-0.5 * lam * delta_z_px ** 2)   # (n)

    # # mean xy from the slice of cov3d given z
    # # (n, 2) + (n, 2) / (n, 1) * (n, 1) -> (n, 2)
    mean_shift = V / (W[:, None] + 1e-8) * delta_z[:, None] # * width / 2   # (n, 2)
    mean_shift = torch.cat([mean_shift, torch.zeros_like(mean_shift[:, :1])], dim=-1)   # (n, 3)
    
    # cov2d from the slice of cov3d given z
    # (n, 2, 1) @ (n, 1, 2) -> (n, 2, 2)
    VV_t = torch.bmm(V[..., None], V[:, None, :])  # (n, 2, 2)

    # (n, 2, 2) - (n, 2, 2) / (n, 1, 1) -> (n, 2, 2)
    cov2d_z = U - VV_t / (W[:, None, None] + 1e-8)

    cov2d_z *= ratio_x_z
    
    return mean_shift, cov2d_z, scale_term