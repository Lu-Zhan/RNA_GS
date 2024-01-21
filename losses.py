import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F
from skimage import filters


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

    simi = (
        simi
        / torch.norm(pred_code, dim=-1, keepdim=True)
        / torch.norm(codebook, dim=-1, keepdim=True).T
    )

    min_dist = 1 - simi.max(dim=-1)[0]  # (num_samples, )
    # min_list = simi.max(dim=-1)[1]

    return min_dist.mean()


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.exp(
            torch.tensor(
                [
                    -((x - window_size // 2) ** 2) / float(2 * sigma**2)
                    for x in range(window_size)
                ]
            )
        )
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

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

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    SSIM_numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    SSIM_denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    SSIM = SSIM_numerator / SSIM_denominator

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def ssim_loss(img1, img2, window_size=7, size_average=True):
    return ssim(img1, img2, window_size, size_average).mean()


# (zwx)
def li_codeloss(pred_code, codebook):
    pred_code_bin = torch.zeros_like(pred_code)
    for i in range(pred_code.shape[1]):
        # 取出同一张图片的alpha
        image = pred_code[:, i]
        image_np = np.asarray(image.detach().cpu())
        best_threshold = filters.threshold_li(image_np)
        # 与阈值进行比较
        binary_image = image > best_threshold
        pred_code_bin[:, i] = binary_image
    expanded_pred_code = pred_code_bin.unsqueeze(1)
    expanded_codebook = codebook.unsqueeze(0)
    hamming_distance = (expanded_pred_code != expanded_codebook).sum(
        dim=2
    ) / codebook.shape[1]
    min_dist = hamming_distance.min(dim=-1)[0]
    min_list = hamming_distance.min(dim=-1)[1]
    return min_dist.mean(), min_dist


def otsu_codeloss(pred_code, codebook):
    pred_code_bin = torch.zeros_like(pred_code)
    for i in range(pred_code.shape[1]):
        # 取出同一张图片的alpha
        image = pred_code[:, i]
        image_np = np.asarray(image.detach().cpu())
        best_threshold = filters.threshold_otsu(image_np)
        # 与阈值进行比较
        binary_image = image > best_threshold
        pred_code_bin[:, i] = binary_image
    expanded_pred_code = pred_code_bin.unsqueeze(1)
    expanded_codebook = codebook.unsqueeze(0)
    hamming_distance = (expanded_pred_code != expanded_codebook).sum(
        dim=2
    ) / codebook.shape[1]
    min_dist = hamming_distance.min(dim=-1)[0]
    min_list = hamming_distance.min(dim=-1)[1]
    return min_dist.mean(), min_dist


def codebook_hamming_loss(pred_code, codebook, mode):
    if mode == "normal":
        threshold = 0.5
    elif mode == "mean":
        sorted_ranks, _ = torch.sort(pred_code, dim=0)
        # 计算每列前70%的阈值
        threshold = sorted_ranks[int(0.7 * pred_code.size(0))]
    elif mode == "median":
        threshold = torch.median(pred_code, dim=0).values
    else:
        print("code loss mode error")
        threshold = 0
    pred_code_bin = torch.where(pred_code > threshold, torch.tensor(1), torch.tensor(0))
    expanded_pred_code = pred_code_bin.unsqueeze(1)
    expanded_codebook = codebook.unsqueeze(0)
    hamming_distance = (expanded_pred_code != expanded_codebook).sum(
        dim=2
    ) / codebook.shape[1]
    min_dist = hamming_distance.min(dim=-1)[0]
    min_list = hamming_distance.min(dim=-1)[1]
    return min_dist.mean(), min_dist


# (lz) compute sigma_x and sigma_y from conics
def obtain_sigma_xy(conics):
    # conics is a tensor of shape (N, 3), inverse covariance matrix in upper triangular format, whose unit is pixel
    # recover to covariance matrix
    inv_cov = torch.stack([conics[:, 0], conics[:, 1], conics[:, 1], conics[:, 2]], dim=1)    # (N, 4)
    inv_cov = inv_cov.view(-1, 2, 2)    # (N, 2, 2)

    cov = torch.inverse(inv_cov)    # (N, 2, 2)

    sigma_x = torch.sqrt(cov[:, 0, 0])    # (N, )
    sigma_y = torch.sqrt(cov[:, 1, 1])    # (N, )

    return sigma_x, sigma_y

#(gzx)
def obtain_sigma_xy_rho(conics):
    # conics is a tensor of shape (N, 3), inverse covariance matrix in upper triangular format, whose unit is pixel
    # recover to covariance matrix
    inv_cov = torch.stack([conics[:, 0], conics[:, 1], conics[:, 1], conics[:, 2]], dim=1)    # (N, 4)
    inv_cov = inv_cov.view(-1, 2, 2)    # (N, 2, 2)

    cov = torch.inverse(inv_cov)    # (N, 2, 2)

    sigma_x = torch.sqrt(cov[:, 0, 0])    # (N, )
    sigma_y = torch.sqrt(cov[:, 1, 1])    # (N, )
    rho     = torch.sqrt((cov[:, 0, 1]*cov[:, 1, 0])/(cov[:, 0, 0]*cov[:, 1, 1]))

    return sigma_x, sigma_y, rho


# (lz) circle loss for 2D gaussian kernal
def circle_loss(sigma_x, sigma_y):
    # the difference betwwen sigma_x and sigma_y should be small
    # sigma_x and sigma_y are tensors of shape (N, )

    return ((sigma_x - sigma_y) ** 2).mean()


# (lz) size loss for 2D gaussian kernal
def size_loss(sigma_x, sigma_y, min_size=6, max_size=12):
    # sigma_x or sigma_y should be in a certain range of [6, 12]

    loss_x = torch.relu(min_size - sigma_x).mean() + torch.relu(sigma_x - max_size).mean()
    loss_y = torch.relu(min_size - sigma_y).mean() + torch.relu(sigma_y - max_size).mean()

    return loss_x + loss_y

#(gzx) calculate lamda1 and lamda2 for 2d gaussian
def rho_loss(sigma_x, sigma_y, rho):
    min_rho = 1
    # rho should be in a range of [min_rho,1]
    
    lamda1_times2 = (sigma_x**2 + sigma_y**2) + torch.sqrt(((sigma_x**2 + sigma_y**2))**2 - 4 * (1 - rho**2) * ((sigma_x * sigma_y)**2))
    lamda2_times2 = (sigma_x**2 + sigma_y**2) - torch.sqrt(((sigma_x**2 + sigma_y**2))**2 - 4 * (1 - rho**2) * ((sigma_x * sigma_y)**2))
    
    dis = torch.relu(min_rho - (lamda1_times2 / lamda2_times2))
    
    return dis.mean()


# (zwx)
def scale_loss(scale_x, scale_y):
    # sigma -> axis length, need modification
    scale_x = torch.sigmoid(scale_x)
    scale_y = torch.sigmoid(scale_y)
    diff_x = scale_x - torch.clamp(scale_x, 0.12, 0.24)
    diff_y = scale_y - torch.clamp(scale_x, 0.12, 0.24)
    scale_loss_x = torch.where(
        torch.abs(diff_x) < 0.5,
        0.5 * diff_x**2,
        0.5 * (torch.abs(diff_x) - 0.5 * 0.5),
    )
    scale_loss_y = torch.where(
        torch.abs(diff_y) < 0.5,
        0.5 * diff_y**2,
        0.5 * (torch.abs(diff_y) - 0.5 * 0.5),
    )
    scale_loss = torch.mean(scale_loss_x) + torch.mean(scale_loss_y)
    return scale_loss


def codebook_loss(type, alpha, codebook, flag, iter,formal_code_loss = 0):
    if type == "cos":
        loss_cos_dist = codebook_cos_loss(alpha, codebook)
        formal_code_loss = 0
    else:
        if flag:
            loss_cos_dist, _ = codebook_hamming_loss(
                alpha, codebook, "normal"
            )
        else:
            if type == "mean":
                loss_cos_dist, _ = codebook_hamming_loss(
                    alpha, codebook, "mean"
                )
            elif type == "median":
                loss_cos_dist, _ = codebook_hamming_loss(
                    alpha, codebook, "median"
                )
            elif type == "li":
                loss_cos_dist, _ = li_codeloss(alpha, codebook)
            elif type == "otsu":
                loss_cos_dist, _ = otsu_codeloss(alpha, codebook)
        if iter == 0:
            formal_code_loss = abs(loss_cos_dist.item())
        elif (
            iter % 200 == 0
            and abs(formal_code_loss - loss_cos_dist) < 0.01
            and flag == True
        ):
            # print(f'start using {self.cfg["cali_loss_type"]} as threshold')
            formal_code_loss = loss_cos_dist.item()
            flag = False
        elif iter % 200 == 0:
            formal_code_loss = loss_cos_dist.item()

        tolerance = 2 / 15.0

        if loss_cos_dist < tolerance:
            loss_cos_dist = 0
        else:
            loss_cos_dist -= tolerance
    
    return loss_cos_dist, flag, formal_code_loss