import os
import wandb
import torch

from PIL import Image
from torch import optim

from lightning import LightningModule
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians

from losses import *
from visualize import view_recon, view_positions
from utils import calculate_mdp_psnr, read_codebook
from models import GaussModel


''' system '''
class GSSystem(LightningModule):
    def __init__(self, hparams, **kwargs):  
        super().__init__()

        self.save_hyperparameters(hparams)
        self.B_SIZE = hparams['train']['tile_size']

        # save folder
        self.save_folder = hparams['exp_dir']
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "epoch"), exist_ok=True)

        self.frames = []

        self.gs_model = GaussModel(
            num_points=self.hparams['train']['num_samples'], 
            hw=self.hparams['hw'],
            device=torch.device(f"cuda:{self.hparams['devices'][0]}"),
        )

        self.codebook = read_codebook(self.hparams['data']['codebook_path'], bg=True)
        self.codebook = torch.tensor(self.codebook, device=self.gs_model.means_3d.device)

        self.dapi_images = kwargs.get('dapi_images', None)
        try:
            self.mdp_dapi_image = self.dapi_images.max(dim=-1)[0]
        except:
            print("dapi_images not available, read from checkpoint")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.gs_model.parameters, lr=self.hparams['train']['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        output, conics, radii, _ = self.forward()

        loss = 0.

        if self.hparams['loss']['w_l1'] > 0:
            loss_l1 = l1_loss(output, batch)
            loss += self.hparams['loss']['w_l1'] * loss_l1
            self.log_step("train/loss_l1", loss_l1)

        if self.hparams['loss']['w_l2'] > 0:
            loss_l2 = mse_loss(output, batch)
            loss += self.hparams['loss']['w_l2'] * loss_l2
            self.log_step("train/loss_l2", loss_l2)
        
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
        
        if self.hparams['loss']['w_rho'] > 0:
            loss_rho = rho_loss(conics)
            loss += self.hparams['loss']['w_rho'] * loss_rho
            self.log_step("train/loss_rho", loss_rho)
        
        if self.hparams['loss']['w_radius'] > 0:
            loss_radius = radius_loss(radii.to(self.gs_model.means_3d.dtype))
            loss += self.hparams['loss']['w_radius'] * loss_radius
            self.log_step("train/loss_radius", loss_radius)
        
        if self.hparams['loss']['w_mi'] > 0:
            pred_code = self.pred_color
            loss_mi = mi_loss(pred_code, self.codebook)

            loss += self.hparams['loss']['w_mi'] * loss_mi
            self.log_step("train/loss_mi", loss_mi)

        self.log_step("train/total_loss", loss, prog_bar=True)
        self.log_step("train/", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        return loss

    def log_step(self, name, loss, on_step=True, on_epoch=False, prog_bar=False):
        self.log(name, loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def validation_step(self, batch, batch_idx):
        batch = batch[0]
        output, _, _, xys = self.forward()

        # metric
        mean_mse = torch.mean((output - batch) ** 2).cpu()
        mean_psnr = float(10 * torch.log10(1 / mean_mse))
        mdp_psnr = calculate_mdp_psnr(output, batch)
        self.log_step('val/mean_psnr', mean_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log_step('val/mdp_psnr', mdp_psnr, on_step=False, on_epoch=True,)

        # visualization
        recon_image = view_recon(
            pred=self._original(output), 
            gt=self._original(batch),
        )
        recon_image = Image.fromarray(recon_image)
        self.frames.append(recon_image)
        recon_image.save(os.path.join(self.save_folder, "epoch", f"epoch_{self.global_step:05d}.png"))

        if self.global_step % 5000 == 0:
            self.logger.experiment.log({"val_image": [wandb.Image(recon_image, caption="val_image")]}, step=self.global_step)
        
        if self.global_step % 10000 == 0:
            position_on_dapi_image = view_positions(
                points_xy=xys.detach().cpu().numpy(), 
                bg_image=self.mdp_dapi_image.cpu().numpy(),
                alpha=self.pred_color.cpu().numpy(),
            )
            position_on_dapi_image.save(os.path.join(self.save_folder, f"positions_dapi.png"))

            position_on_mdp_image = view_positions(
                points_xy=xys.detach().cpu().numpy(), 
                bg_image=batch.max(dim=-1)[0].cpu().numpy(),
                alpha=self.pred_color.cpu().numpy(),
            )
            position_on_mdp_image.save(os.path.join(self.save_folder, f"positions_mdp.png"))

            self.logger.experiment.log({"positions": [
                wandb.Image(position_on_dapi_image, caption="dapi"),
                wandb.Image(position_on_mdp_image, caption="mdp"),
            ]})

        if self.hparams['model']['save_gif'] == 1 and self.global_step == self.hparams['train']['iterations']:
            frames = [Image.fromarray(x) for x in self.frames]
            save_path = os.path.join(self.save_folder, "training.gif")
            frames[0].save(save_path, save_all=True, append_images=frames[1:], optimize=False, duration=5, loop=0)

        return mean_psnr

    def predict_step(self, batch, batch_idx):
        batch = batch[0]
        output, _, _, xys = self.forward()

        # metric
        mean_mse = torch.mean((output - batch) ** 2).cpu()
        mean_psnr = float(10 * torch.log10(1 / mean_mse))
        mdp_psnr = calculate_mdp_psnr(output, batch)
        print(f'mean_psnr: {mean_psnr:.4f}, mdp_psnr: {mdp_psnr:.4f}')

        # visualization
        recon_image = view_recon(
            pred=self._original(output), 
            gt=self._original(batch),
            resize=(384, 384),
        )
        recon_image = Image.fromarray(recon_image)

        recon_image.save(os.path.join(self.save_folder, f"recon.png"))

        position_on_dapi_image = view_positions(
            points_xy=xys.detach().cpu().numpy(), 
            bg_image=self.mdp_dapi_image.cpu().numpy(),
            alpha=self.pred_color.cpu().numpy(),
        )
        position_on_dapi_image.save(os.path.join(self.save_folder, f"positions_dapi.png"))

        position_on_mdp_image = view_positions(
            points_xy=xys.detach().cpu().numpy(), 
            bg_image=batch.max(dim=-1)[0].cpu().numpy(),
            alpha=self.pred_color.cpu().numpy(),
        )
        position_on_mdp_image.save(os.path.join(self.save_folder, f"positions_mdp.png"))
        
        try:
            self.logger.experiment.log({"positions": [
                wandb.Image(position_on_dapi_image, caption="dapi"),
                wandb.Image(position_on_mdp_image, caption="mdp"),
            ]})
        except:
            print("wandb not available")

        return None

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

        return out_img, conics, radii, xys

    def on_save_checkpoint(self, checkpoint):
        checkpoint['gs_model'] = self.gs_model
        checkpoint['mdp_dapi_image'] = self.mdp_dapi_image

    def on_load_checkpoint(self, checkpoint):
        self.gs_model = checkpoint['gs_model']
        self.mdp_dapi_image = checkpoint['mdp_dapi_image']

    def _original(self, x):
        return x * (self.hparams['value_range'][1] - self.hparams['value_range'][0]) + self.hparams['value_range'][0]
    
    @property
    def pred_color(self):
        return torch.sigmoid(self.gs_model.rgbs) * torch.sigmoid(self.gs_model.opacities)
