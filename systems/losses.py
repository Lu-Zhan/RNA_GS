import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F
from skimage import filters


l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
sl1_loss = torch.nn.SmoothL1Loss()


def mi_loss(pred_code, codebook, min_value=0):
    pred_code = torch.cat([pred_code, torch.ones_like(pred_code[:, :1])], dim=-1)  # (num_samples, 16)
    codebook = torch.cat([codebook, torch.ones_like(codebook[:, :1])], dim=-1)  # (181, 16)
    
    simi = pred_code @ codebook.T  # (num_samples, 16) @ (16, 181) = (num_samples, 181)

    simi = simi / (torch.norm(pred_code, dim=-1, keepdim=True) * torch.norm(codebook, dim=-1, keepdim=True).T + 1e-8)

    # all_dist = 1 - simi # (num_samples, 15)
    # min_dist = all_dist.min(dim=-1, keepdim=True)[0]  # (num_samples, 1)

    all_dist = simi # (num_samples, 15)
    min_dist = all_dist.max(dim=-1, keepdim=True)[0]  # (num_samples, 1)

    exp_all_dist = torch.exp(all_dist).sum(dim=-1, keepdim=True)
    exp_min_dist = torch.exp(min_dist)

    loss = - torch.log(exp_min_dist / (exp_all_dist - exp_min_dist + 1e-8)) - min_value

    return loss.mean()


def obtain_simi(pred_code, codebook):
    simi = pred_code @ codebook.T  # (num_samples, 15) @ (15, 181) = (num_samples, 181)
    simi = simi / (torch.norm(pred_code, dim=-1, keepdim=True) * torch.norm(codebook, dim=-1, keepdim=True).T + 1e-8)
    max_simi = simi.max(dim=-1)[0]  # (num_samples, )
    index = simi.max(dim=-1)[1]

    return max_simi, index


def cos_loss(pred_code, codebook):
    max_simi = obtain_simi(pred_code, codebook)[0]
    
    return 1 - max_simi.mean()


def masked_l1_loss(input, target, th=0.05):
    mask = target > th

    return (input * mask - target * mask).abs().sum() / (mask.sum() + 1e-8)


def masked_mse_loss(input, target, th=0.05):
    mask = target > th

    return ((input * mask - target * mask) ** 2).sum() / (mask.sum() + 1e-8)


def bg_l1_loss(input, target, th=0.05):
    mask = target < th
    return ((input * mask).abs()).sum() / (mask.sum() + 1e-8)


def bg_mse_loss(input, target, th=0.05):
    mask = target < th
    return ((input * mask) ** 2).sum() / (mask.sum() + 1e-8)


def rho_loss(cov2d):
     # cov2d, (n, 3)
    rho_2 = cov2d[:, 1] ** 2 / (cov2d[:, 0] * cov2d[:, 2] + 1e-8)   

    return rho_2.mean()


def radius_loss(radii, range=[2, 7]):
    # loss_small = (torch.relu(range[0] - radii) ** 2).mean()
    loss_big = (torch.relu(radii - range[0]) ** 2).mean()

    return loss_big


def calculate_mdp_psnr(img, gt_img):
    MDP_img = img.max(axis=2).values
    MDP_gt_img = gt_img.max(axis=2).values
    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(MDP_img, MDP_gt_img)
    mdp_psnr = float(10 * torch.log10(1 / mse))
    return mdp_psnr


if __name__ == '__main__':
    codebook = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 0, 0],
    ], dtype=torch.float32)

    pred_code = torch.tensor([
        [0, 0, 0, 0],
        [0.1, 0, 0, 0],
        [0.2, 0, 0, 0],
        [0.4, 0, 0, 0],
        [0.7, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 0.1],
        [1, 1, 1, 0.2],
        [1, 1, 1, 0.4],
        [1, 1, 1, 0.7],
        [1, 1, 1, 1],
    ], dtype=torch.float32)

    codebook = torch.cat([codebook, torch.zeros(codebook.shape[0], 180)], dim=-1)
    pred_code = torch.cat([pred_code, torch.zeros(pred_code.shape[0], 180)], dim=-1)
                         
    min_value = - np.log(np.exp(1) / (codebook.shape[-1] + 1e-8))
    print(min_value)
    print(mi_loss(pred_code, codebook, min_value=0))

