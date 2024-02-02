import math
import os
import gc
import time
import torch
import wandb
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional
from torch import Tensor, optim
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

def get_b(sigma_x, sigma_y):
    return (sigma_x**2 + sigma_y**2) / 2

def get_det_and_inv(sigma_x, sigma_y, rho):
    
    det = (1 - rho**2)*(sigma_x**2)*(sigma_y**2)
    inv_det = 1.0 / det    
    
    return det,inv_det

def get_lambda(sigma_x, sigma_y, rho):
    
    b = get_b(sigma_x,sigma_y)
    det,inv_det = get_det_and_inv(sigma_x, sigma_y, rho)
    
    lambda1 = b + torch.sqrt(torch.clamp(b**2 - det,min=0.0001))
    lambda2 = b - torch.sqrt(torch.clamp(b**2 - det,min=0.0001))
    
    return lambda1,lambda2

def get_radii(sigma_x, sigma_y, rho):
    
    lambda1, lambda2 = get_lambda(sigma_x, sigma_y, rho)
    radii = (torch.sqrt(3 * lambda1)).ceil().int()
    
    return radii

def get_conics(sigma_x, sigma_y, rho):
    
    det,inv_det = get_det_and_inv(sigma_x, sigma_y, rho)    
    conics = torch.cat([
        inv_det * (sigma_y**2),
        inv_det * (-rho * sigma_y * sigma_x),
        inv_det * (sigma_x**2),
        ],dim=-1)    
    
    return conics
    
def project_gaussians_2D(sigma_x, sigma_y, rho, num_points, depth, device):
    
    depths = torch.ones(num_points, device=device) * depth
    radii = get_radii(sigma_x, sigma_y, rho)
    conics = get_conics(sigma_x, sigma_y, rho)
    num_tiles_hit = torch.ones(num_points,dtype = torch.int32, device=device) * 4
    
    return depths, radii, conics, num_tiles_hit
    
    