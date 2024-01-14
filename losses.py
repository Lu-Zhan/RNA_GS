import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F


l1_loss = torch.nn.L1Loss()
mse_loss = torch.nn.MSELoss()
sl1_loss = torch.nn.SmoothL1Loss()

def masked_l1_loss(input, target):
    mask = target > 0

    return (input * mask - target * mask).abs().sum() / (mask.sum() + 1e-8)

def masked_mse_loss(input, target):
    mask = target > 0

    return ((input * mask - target * mask) ** 2).sum() / (mask.sum() + 1e-8)


def bg_loss(input, target):
    mask = target == 0
    return ((input * mask) ** 2).sum() / (mask.sum() + 1e-8)


def bright_loss(input, target):
    mask_input = input > 0
    mask_target = target > 0

    return (mask_input - mask_target).mean()


def codebook_cos_loss(pred_code, codebook):
    simi = pred_code @ codebook.T  # (num_samples, 15) @ (15, 181) = (num_samples, 181)

    simi = simi / torch.norm(pred_code, dim=-1, keepdim=True) / torch.norm(codebook, dim=-1, keepdim=True).T

    min_dist = 1 - simi.max(dim=-1)[0]  # (num_samples, )
    # min_list = simi.max(dim=-1)[1]

    return min_dist.mean()


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, window_size=11, size_average=True):
    # Assuming the image is of shape [N, C, H, W]
    (_, _, channel) = img1.size()

    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)


    # Parameters for SSIM
    C1 = 0.01**2
    C2 = 0.03**2

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    SSIM_numerator = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def ssim_loss(img1, img2, window_size=7, size_average=True):
    return ssim(img1, img2, window_size, size_average).mean()
