import os
import wandb
import torch

from PIL import Image
from torch import optim

from lightning import LightningModule
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from losses import *
from visualize import view_output
from utils import calculate_mdp_psnr
from models import GaussModel


''' system '''
class GSSystem(LightningModule):
    def __init__(self, hparams):  
        super().__init__()

        self.save_hyperparameters(hparams)
        self.B_SIZE = hparams['train']['tile_size']

        # save folder
        self.save_folder = hparams['exp_dir']
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "epoch"), exist_ok=True)

        self.frames = []
    
    def prepare_data(self):
        self.gs_model = GaussModel(
            num_points=self.hparams['train']['num_samples'], 
            hw=self.hparams['hw'],
            device=torch.device(f"cuda:{self.hparams['devices'][0]}"),
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.gs_model.parameters, lr=self.hparams['train']['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        output = self.forward()

        loss = 0.

        if self.hparams['loss']['w_l1'] > 0:
            loss_l1 = l1_loss(output, batch)
            loss += self.hparams['loss']['w_l1'] * loss_l1
            self.log_step("train/loss_l1", loss_l1)

        if self.hparams['loss']['w_l2'] > 0:
            loss_l2 = mse_loss(output, batch)
            loss += self.hparams['loss']['w_l2'] * loss_l2
            self.log_step("train/loss_l2", loss_l2)
        
        if self.hparams['loss']['w_mi'] > 0:
            loss_mi = mi_loss(output, batch)
            loss += self.hparams['loss']['w_mi'] * loss_mi
            self.log_step("train/loss_mi", loss_l1)
        
        if self.hparams['loss']['w_masked_l2'] > 0:
            loss_masked_l2 = masked_mse_loss(output, batch)
            loss += self.hparams['loss']['w_masked_l2'] * loss_masked_l2
            self.log_step("train/loss_masked_l2", loss_masked_l2)
        
        if self.hparams['loss']['w_bg_l2'] > 0:
            loss_bg_l2 = bg_mse_loss(output, batch)
            loss += self.hparams['loss']['w_bg_l2'] * loss_bg_l2
            self.log_step("train/loss_bg_l2", loss_bg_l2)
        
        if self.hparams['loss']['w_masked_l1'] > 0:
            loss_masked_l1 = masked_l1_loss(output, batch)
            loss += self.hparams['loss']['w_masked_l1'] * loss_masked_l1
            self.log_step("train/loss_masked_l1", loss_masked_l1)
        
        if self.hparams['loss']['w_bg_l1'] > 0:
            loss_bg_l1 = bg_l1_loss(output, batch)
            loss += self.hparams['loss']['w_bg_l1'] * loss_bg_l1
            self.log_step("train/loss_bg_l1", loss_bg_l1)

        self.log_step("train/total_loss", loss, prog_bar=True)

        return loss

    def log_step(self, name, loss, on_step=True, on_epoch=False, prog_bar=False):
        self.log(name, loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        output = self.forward()

        # metric
        mean_mse = torch.mean((output - batch) ** 2).cpu()
        mean_psnr = float(10 * torch.log10(1 / mean_mse))
        mdp_psnr = calculate_mdp_psnr(output, batch)
        self.log_step('val/mean_psnr', mean_psnr, prog_bar=True)
        self.log_step('val/mdp_psnr', mdp_psnr)

        # visualization
        view = view_output(pred=output, gt=batch)
        view = Image.fromarray(view)
        self.frames.append(view)

        view.save(os.path.join(self.save_folder, "epoch", f"epoch_{self.global_step:05d}.png"))
        self.logger.experiment.log({"val_image": [wandb.Image(view, caption="val_image")]}, step=self.global_step)

        if self.hparams['model']['save_gif'] == 1 and self.global_step == self.hparams['train']['iterations']:
            frames = [Image.fromarray(x) for x in self.frames]
            save_path = os.path.join(self.save_folder, "training.gif")
            frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=False, duration=5, loop=0)

    def forward(self):
        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
            self.gs_model.means_3d,
            self.gs_model.scales,
            1,
            self.gs_model.quats,
            self.gs_model.viewmat,
            self.gs_model.viewmat,
            self.gs_model.focal,
            self.gs_model.focal,
            self.gs_model.W / 2,
            self.gs_model.H / 2,
            self.gs_model.H,
            self.gs_model.W,
            self.B_SIZE,
        )
        # torch.cuda.synchronize()
       
        out_img = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(self.gs_model.rgbs),
            torch.sigmoid(self.gs_model.opacities),
            self.gs_model.H,
            self.gs_model.W,
            self.B_SIZE,
            self.gs_model.background,
        )
        # torch.cuda.synchronize()

        return out_img




