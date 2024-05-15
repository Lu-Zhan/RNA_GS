import os
import wandb
import torch

from PIL import Image
from torch import optim

from lightning import LightningModule

from systems.losses import *
from utils.vis2d_utils import view_recon
from utils.utils import read_codebook
from systems.models import model_zoo
from systems.cameras import SliceCameras

from systems.recon_systems import *


class GSRegSystem(GSSystem3D):
    def setup(self, stage):
        self.gs_model.require_grad(False)
    
    def configure_optimizers(self):
        cam_optimizer = optim.Adam(self.cam_model.parameters, lr=self.hparams['train']['lr_cam'])
        return cam_optimizer

    def training_step(self, batch, batch_idx):
        gt_3d, cam_indexs, slice_indexs, reg_masks = batch

        recon_3d, cov2d, radii, _ = self.gs_model.render_slice(
            camera=self.cam_model, cam_indexs=cam_indexs, slice_indexs=slice_indexs,
        )

        # maskout unregistered regions
        gt_3d[reg_masks == 0] = recon_3d[reg_masks == 0]

        gt_2d = gt_3d.max(dim=0)[0]    # (k, h, w, c) -> (h, w, c)
        gt_mdp = gt_2d.max(dim=-1)[0]  # (h, w, c) -> (h, w)

        recon_2d = recon_3d.max(dim=0)[0]  #.max(dim=0)[0]   # (n, h, w, k) -> (h, w, k)
        recon_mdp = recon_2d.max(dim=-1)[0]  #.max(dim=0)[0]   # (h, w, k) -> (h, w)

        results = {
            '3d': (recon_3d, gt_3d),
            '2d': (recon_2d, gt_2d),
            'mdp': (recon_mdp, gt_mdp),
        }

        # loss = self.compute_loss_with_mask(results, reg_masks)
        loss = self.compute_loss(results)
        
        self.log_step("train/total_loss", loss, prog_bar=True)
        self.logger.experiment.log({
            f"cam_zs/{idx}": self.cam_model.camera_zs[idx] for idx in range(self.hparams['camera']['max_num_cams'])
        })
        if self.hparams['train']['refine_camera']:
            self.log_step("params/lr_cam", self.trainer.optimizers[1].param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        batch, cam_indexs, slice_indexs, _ = batch
        self.validation_step_outputs.append((self.obtain_output(cam_indexs, slice_indexs), batch))

    def forward_evaluation(self, recon_3d, gt_3d, xys, is_predict=False, log_each_plane=False):
        recon_3d = self._original(recon_3d)
        gt_3d = self._original(gt_3d)

        # metric
        mean_3d_mse = torch.mean((recon_3d - gt_3d) ** 2).cpu()
        mean_3d_psnr = float(10 * torch.log10(1 / mean_3d_mse))

        recon_2d = recon_3d.max(dim=0)[0]
        gt_2d = gt_3d.max(dim=0)[0]

        mean_2d_mse = torch.mean((recon_2d - gt_2d) ** 2).cpu()
        mean_2d_psnr = float(10 * torch.log10(1 / mean_2d_mse))

        mdp_psnr = calculate_mdp_psnr(recon_2d, gt_2d)

        if not is_predict:
            self.log_step('val/mean_3d_psnr', mean_3d_psnr, on_step=False, on_epoch=True, prog_bar=True)
            self.log_step('val/mean_2d_psnr', mean_2d_psnr,  on_step=False, on_epoch=True)
            self.log_step('val/mdp_psnr', mdp_psnr, on_step=False, on_epoch=True)

        # visualization
        recon_images = view_recon(pred=recon_2d, gt=gt_2d)[0]
        recon_images = Image.fromarray(recon_images)
        recon_images.save(os.path.join(self.save_folder, "recon", f"iter_{self.global_step:05d}.png"))
        self.logger.experiment.log({"val_image": [wandb.Image(recon_images, caption="val_image")]})
        
        # visualization
        vmax = float(gt_2d.data.ravel().max())
        vmin = float(gt_2d.ravel()[gt_2d.ravel() > 0].min()) if (gt_2d.ravel() > 0).sum() > 0 else 0
        recon_images = [Image.fromarray(view_recon(pred=x, gt=y, vmax=vmax, vmin=vmin)[0]) for x, y in zip(recon_3d, gt_3d)]

        for i, image in enumerate(recon_images):
            image.save(os.path.join(self.save_folder, "recon_plane", f"iter_{self.global_step:05d}_{i}.png"))
        
        if log_each_plane:
            self.logger.experiment.log({
                "recon_plane": [
                    wandb.Image(x, caption=f"plane_{i}") for i, x in enumerate(recon_images)
                ],
            })

        return mean_3d_psnr
        