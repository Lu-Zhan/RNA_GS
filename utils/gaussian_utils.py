import torch

import torch.nn.functional as F

from einops import rearrange, repeat
from gsplat._torch_impl import (
    get_tile_bbox, 
    project_pix, 
    compute_cov2d_bounds, 
    clip_near_plane, 
    scale_rot_to_cov3d, 
    # ndc2pix,
)


def slice_3d_gaussians_by_single_z(
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

    mean_shift = mean_shift[..., 0]  # (n, 3)
    scale_term = scale_term[..., 0]  # (n,)

    # shift xy at plane given z, depth is not change!
    means3d = means3d + mean_shift

    # (n, 3), (n,), (n,)
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
        num_tiles_hit,
        mask,
        scale_term,
    )


def slice_3d_gaussians_by_multi_z(
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

    # (n, 3, k), (n, 2, 2), (n, k)
    mean_shift, cov2d, scale_term = obtain_cov2d_and_mean_shift(
        means3d, cov3d, plane_z, torch.abs(viewmat[2, 3])
    )

    # shift xy at plane given z, depth is not change!
    means3d = means3d[..., None] + mean_shift # (n, 3, k)

    means3d = rearrange(means3d, 'n c k -> (n k) c')  # (n * k, 3)

    # (n, 3), (n,), (n,)
    conic, radius, det_valid = compute_cov2d_bounds(cov2d)
    xys = project_pix(fullmat, means3d, img_size, (cx, cy)) # (n * k, 2)

    conic = repeat(conic, 'n c -> (n k) c', k=mean_shift.shape[-1])  # (n * k, 3)
    radius = repeat(radius, 'n -> (n k)', k=mean_shift.shape[-1])  # (n * k,)
    det_valid = repeat(det_valid, 'n -> (n k)', k=mean_shift.shape[-1])  # (n * k,)
    cov3d = repeat(cov3d, 'n c d -> (n k) c d', k=mean_shift.shape[-1])  # (n * k, 3, 3)
    cov2d = repeat(cov2d, 'n c d -> (n k) c d', k=mean_shift.shape[-1])  # (n * k, 2, 2)
    p_view = repeat(p_view, 'n c -> (n k) c', k=mean_shift.shape[-1])  # (n * k, 3)

    tile_min, tile_max = get_tile_bbox(xys, radius, tile_bounds, block_width)
    tile_area = (tile_max[..., 0] - tile_min[..., 0]) * (
        tile_max[..., 1] - tile_min[..., 1]
    )

    is_close = repeat(is_close, 'n -> (n k)', k=mean_shift.shape[-1])  # (n * k,)
    mask = (tile_area > 0) & (~is_close) & det_valid

    num_tiles_hit = tile_area   # (n * k,)
    depths = p_view[..., 2]
    radii = radius.to(torch.int32)

    radii = torch.where(~mask, 0, radii)
    conic = torch.where(~mask[..., None], 0, conic)
    xys = torch.where(~mask[..., None], 0, xys)
    cov3d = torch.where(~mask[..., None, None], 0, cov3d)
    cov2d = torch.where(~mask[..., None, None], 0, cov2d)
    num_tiles_hit = torch.where(~mask, 0, num_tiles_hit)
    depths = torch.where(~mask, 0, depths)

    i, j = torch.triu_indices(3, 3)
    cov3d_triu = cov3d[..., i, j]
    i, j = torch.triu_indices(2, 2)
    cov2d_triu = cov2d[..., i, j]

    cov3d_triu = rearrange(cov3d_triu, '(n k) d -> n d k', k=mean_shift.shape[-1])
    cov2d_triu = rearrange(cov2d_triu, '(n k) d -> n d k', k=mean_shift.shape[-1])
    xys = rearrange(xys, '(n k) c -> n c k', k=mean_shift.shape[-1])
    depths = rearrange(depths, '(n k) -> n k', k=mean_shift.shape[-1])
    radii = rearrange(radii, '(n k) -> n k', k=mean_shift.shape[-1])
    conic = rearrange(conic, '(n k) c -> n c k', k=mean_shift.shape[-1])
    num_tiles_hit = rearrange(num_tiles_hit, '(n k) -> n k', k=mean_shift.shape[-1])
    mask = rearrange(mask, '(n k) -> n k', k=mean_shift.shape[-1])

    return (
        cov3d_triu,
        cov2d_triu,
        xys,
        depths,
        radii,
        conic,
        num_tiles_hit,
        mask,
        scale_term,
    )


# @torch.compile(mode='max-autotune')
def obtain_cov2d_and_mean_shift(
        means_3d, cov3d, plane_z, ratio_x_z,
    ):

    # plane_z: (k)

    U = cov3d[:, :2, :2]    # (n, 2, 2)
    V = cov3d[:, :2, 2] # (n, 2)
    W = cov3d[:, 2, 2][:, None] # (n, 1)

    plane_z = plane_z.reshape(1, -1)   # (1, k)

    delta_z = plane_z - means_3d[:, -1:]   # (n, k)
    delta_z_px = delta_z * ratio_x_z   # (n, k)

    lam = 1 / (W + 1e-8)    # (n, 1)
    scale_term = torch.exp(-0.5 * lam * delta_z_px ** 2)   # (n, k)

    # mean xy from the slice of cov3d given z
    # (n, 2, 1) / (n, 1, 1) * (n, 1, k) -> (n, 2)
    mean_shift = V[..., None] / (W[..., None] + 1e-8) * delta_z[:, None, :] # * width / 2   # (n, 2, k)
    mean_shift = torch.cat([mean_shift, torch.zeros_like(mean_shift[:, :1, :])], dim=-2)   # (n, 3, k)
    
    # cov2d from the slice of cov3d given z
    # (n, 2, 1) @ (n, 1, 2) -> (n, 2, 2)
    VV_t = torch.bmm(V[..., None], V[:, None, :])  # (n, 2, 2)

    # (n, 2, 2) - (n, 2, 2) / (n, 1, 1) -> (n, 2, 2)
    cov2d_z = U - VV_t / (W[..., None] + 1e-8)
    cov2d_z *= ratio_x_z
    
    # (n, 3, k), (n, 2, 2), (n, k)
    return mean_shift, cov2d_z, scale_term

