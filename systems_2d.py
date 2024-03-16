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
from models import model_zoo


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
        self.mdp_image = None

        self.gs_model = model_zoo[hparams['train']['model']](
            num_primarys=self.hparams['train']['num_primarys'],
            num_backups=self.hparams['train']['num_backups'],
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

        # self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = optim.Adam(self.gs_model.parameters, lr=self.hparams['train']['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        batch = batch[0]

        if self.mdp_image is None:
            self.mdp_image = batch.max(dim=-1)[0]

        den_interval = self.hparams['train']['densification_interval']
        den_start = self.hparams['train']['densification_start']

        if den_interval > 0 and self.hparams['train']['model'] == 'guass':
            if self.global_step % (den_interval + 1) == 0 and self.global_step > den_start:
                self.prune_points()

        output, conics, radii, _ = self.forward()
        mdp_output = output.max(dim=-1)[0]

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
            
        # losses for map image
        if self.hparams['loss']['w_mdp_l2'] > 0:
            loss_mdp_l2 = mse_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_l2'] * loss_mdp_l2
            self.log_step("train/loss_mdp_l2", loss_mdp_l2)
        
        if self.hparams['loss']['w_mdp_masked_l2'] > 0:
            loss_mdp_masked_l2 = masked_mse_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_masked_l2'] * loss_mdp_masked_l2
            self.log_step("train/loss_mdp_masked_l2", loss_mdp_masked_l2)
        
        if self.hparams['loss']['w_mdp_bg_l2'] > 0:
            loss_mdp_bg_l2 = bg_mse_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_bg_l2'] * loss_mdp_bg_l2
            self.log_step("train/loss_mdp_bg_l2", loss_mdp_bg_l2)
        
        if self.hparams['loss']['w_mdp_l1'] > 0:
            loss_mdp_l1 = l1_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_l1'] * loss_mdp_l1
            self.log_step("train/loss_mdp_l1", loss_mdp_l1)

        if self.hparams['loss']['w_mdp_masked_l1'] > 0:
            loss_mdp_masked_l1 = masked_l1_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_masked_l1'] * loss_mdp_masked_l1
            self.log_step("train/loss_mdp_masked_l1", loss_mdp_masked_l1)
        
        if self.hparams['loss']['w_mdp_bg_l1'] > 0:
            loss_mdp_bg_l1 = bg_l1_loss(mdp_output, self.mdp_image)
            loss += self.hparams['loss']['w_mdp_bg_l1'] * loss_mdp_bg_l1
            self.log_step("train/loss_mdp_bg_l1", loss_mdp_bg_l1)
        
        pred_code = self.gs_model.colors
        loss_mi = mi_loss(pred_code, self.codebook)
        self.log_step("train/loss_mi", loss_mi)

        loss_cos = cos_loss(pred_code, self.codebook)
        self.log_step("train/loss_cos", loss_cos)

        if self.hparams['loss']['w_mi'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
            loss += self.hparams['loss']['w_mi'] * loss_mi

        if self.hparams['loss']['w_cos'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
            loss += self.hparams['loss']['w_cos'] * loss_cos
        
        self.log_step("train/total_loss", loss, prog_bar=True)
        self.log_step("params/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log_step("params/num_samples", self.gs_model.current_num_samples, prog_bar=True)

        # self.gs_model.maskout_grad()
        # opt = self.optimizers()
        # opt.zero_grad()

        # self.manual_backward(loss)
        # opt.step()

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
                alpha=self.gs_model.colors.cpu().numpy(),
            )
            position_on_dapi_image.save(os.path.join(self.save_folder, f"positions_dapi.png"))

            position_on_mdp_image = view_positions(
                points_xy=xys.detach().cpu().numpy(),
                bg_image=batch.max(dim=-1)[0].cpu().numpy(),
                alpha=self.gs_model.colors.cpu().numpy(),
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
            alpha=self.gs_model.colors.cpu().numpy(),
        )
        position_on_dapi_image.save(os.path.join(self.save_folder, f"positions_dapi.png"))

        position_on_mdp_image = view_positions(
            points_xy=xys.detach().cpu().numpy(), 
            bg_image=batch.max(dim=-1)[0].cpu().numpy(),
            alpha=self.gs_model.colors.cpu().numpy(),
        )
        position_on_mdp_image.save(os.path.join(self.save_folder, f"positions_mdp.png"))
        
        try:
            self.logger.experiment.log({"positions": [
                wandb.Image(position_on_dapi_image, caption="dapi"),
                wandb.Image(position_on_mdp_image, caption="mdp"),
            ]})
        except:
            print("wandb not available")

        return 1

    def prune_points(self):
        # find indices to remove and update the persistent mask
        rgbs = torch.sigmoid(self.gs_model.rgbs)
        opacities = torch.sigmoid(self.gs_model.opacities)
        colors = rgbs * opacities

        indices_to_remove = (colors.max(dim=-1)[0] < self.hparams['train']['densification_th']).nonzero(as_tuple=True)[0]
        self.gs_model.maskout(indices_to_remove)

    def forward(self):
        means_3d, scales, quats, rgbs, opacities = self.gs_model.obtain_data()

        xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = project_gaussians(
            means_3d,
            scales,
            1,
            quats,
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

        out_img = rasterize_gaussians(
            xys, 
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(rgbs),
            torch.sigmoid(opacities),
            self.gs_model.H,
            self.gs_model.W,
            self.B_SIZE,
            self.gs_model.background,
        )

        return out_img, conics, radii, xys

    def on_save_checkpoint(self, checkpoint):
        checkpoint['gs_model'] = self.gs_model
        checkpoint['mdp_dapi_image'] = self.mdp_dapi_image

    def on_load_checkpoint(self, checkpoint):
        self.gs_model = checkpoint['gs_model']
        self.mdp_dapi_image = checkpoint['mdp_dapi_image']

    def _original(self, x):
        return x * (self.hparams['value_range'][1] - self.hparams['value_range'][0]) + self.hparams['value_range'][0]

        