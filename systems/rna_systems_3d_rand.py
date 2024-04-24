import os
import wandb
import torch

from PIL import Image
from torch import optim

from lightning import LightningModule

from systems.losses import *
from utils.vis2d_utils import view_recon
from utils.utils import read_codebook
from systems.models_3d import model_zoo
from systems.cameras import SliceCamera

import time


class GSSystem3DRand(LightningModule):
    def __init__(self, hparams, **kwargs):  
        super().__init__()
        self.save_hyperparameters(hparams)

        # save folder
        self.save_folder = hparams['exp_dir']
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "recon"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "recon_plane"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, 'classes'), exist_ok=True)

        self.mdp_image = None
        self.is_init_rgb = False

        self.cam_model = SliceCamera(
            num_slice=self.hparams['model']['num_slice'],
            hw=self.hparams['hw'],
            step_z=self.hparams['model']['step_z'],
            camera_z=self.hparams['model']['camera_z'],
            refine_camera=self.hparams['train']['refine_camera'],
            num_dims=self.hparams['model']['num_dims'],
            device=torch.device(f"cuda:{self.hparams['devices'][0]}"),
        )

        self.gs_model = model_zoo[hparams['train']['model']](
            num_primarys=self.hparams['train']['num_primarys'],
            num_backups=self.hparams['train']['num_backups'],
            device=torch.device(f"cuda:{self.hparams['devices'][0]}"),
            camera=self.cam_model,
            B_SIZE=hparams['train']['tile_size'],
        )

        if self.hparams['data']['codebook_path'] != '':
            self.rna_class, self.rna_name = read_codebook(path=self.hparams['data']['codebook_path'], bg=False)
            self.mi_rna_class, self.mi_rna_name = read_codebook(path=self.hparams['data']['codebook_path'], bg=True)
        else:
            # self.rna_class, self.rna_name = np.array([[1., 1., 1.]]), np.array(['full'])
            # self.mi_rna_class, self.mi_rna_name = np.array([[1., 1., 1.], [0., 0., 0.]]), np.array(['full', 'background'])

            self.rna_class, self.rna_name = np.array([[1.]]), np.array(['full'])
            self.mi_rna_class, self.mi_rna_name = np.array([[1.], [0.]]), np.array(['full', 'background'])

        self.rna_class = torch.tensor(self.rna_class, device=self.gs_model.means_3d.device, dtype=self.gs_model.means_3d.dtype)
        self.mi_rna_class = torch.tensor(self.mi_rna_class, device=self.gs_model.means_3d.device, dtype=self.gs_model.means_3d.dtype)

        self.dapi_images = kwargs.get('dapi_images', None)
        try:
            self.mdp_dapi_image = self.dapi_images.max(dim=-1)[0].max(dim=0)[0]
        except:
            print("dapi_images not available, read from checkpoint")

        self.validation_step_outputs = []
        self.predict_step_outputs = []

        # self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = optim.Adam(self.gs_model.parameters, lr=self.hparams['train']['lr'])
        return optimizer

    def training_step(self, batch, batch_idx):
        # batch = batch[0]
        batch, index = batch

        if self.is_init_rgb is False and self.hparams['train']['init_rgb']:
            with torch.no_grad():
                _, _, _, xys = self.gs_model.render(camera=self.cam_model)
                self.gs_model.init_rgbs(xys=xys, gt_images=batch.max(dim=0)[0])
            self.is_init_rgb = True

        if self.mdp_image is None:
            self.mdp_image = batch.max(dim=-1)[0]   #.max(dim=0)[0]

        den_interval = self.hparams['train']['densification_interval']
        den_start = self.hparams['train']['densification_start']

        if den_interval > 0 and self.hparams['train']['model'] == 'guass':
            if self.global_step % (den_interval + 1) == 0 and self.global_step > den_start:
                self.prune_points()

        output, conics, radii, _ = self.gs_model.render_slice(camera=self.cam_model, index=index)

        # output = mdp_slices.max(dim=0)[0]   # (n, h, w, k) -> (h, w, k)
        mdp_output = output.max(dim=-1)[0]  #.max(dim=0)[0]   # (n, h, w, k) -> (h, w)

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

        # if self.hparams['loss']['w_rho'] > 0:
        #     loss_rho = rho_loss(conics)
        #     loss += self.hparams['loss']['w_rho'] * loss_rho
        #     self.log_step("train/loss_rho", loss_rho)
        
        # if self.hparams['loss']['w_radius'] > 0:
        #     loss_radius = radius_loss(radii.to(self.gs_model.means_3d.dtype))
        #     loss += self.hparams['loss']['w_radius'] * loss_radius
        #     self.log_step("train/loss_radius", loss_radius)
            
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
        
        # pred_code = self.gs_model.colors
        # loss_mi = mi_loss(pred_code, self.mi_rna_class)
        # self.log_step("train/loss_mi", loss_mi)

        # loss_cos = cos_loss(pred_code, self.rna_class)
        # self.log_step("train/loss_cos", loss_cos)

        # if self.hparams['loss']['w_mi'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
        #     loss += self.hparams['loss']['w_mi'] * loss_mi

        # if self.hparams['loss']['w_cos'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
        #     loss += self.hparams['loss']['w_cos'] * loss_cos
        

        self.log_step("train/total_loss", loss, prog_bar=True)
        self.log_step("params/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log_step("params/num_samples", self.gs_model.current_num_samples)

        return loss
    
    def obtain_output(self, index):
        output, _, _, xys = self.gs_model.render_slice(camera=self.cam_model, index=index)

        return output, xys

    def forward_evaluation(self, output, batch, xys, is_predict=False, log_each_plane=False):

        output = self._original(output)
        batch = self._original(batch)

        # metric
        mean_mse = torch.mean((output - batch) ** 2).cpu()
        mean_psnr = float(10 * torch.log10(1 / mean_mse))

        mdp_output = output.max(dim=0)[0]
        mdp_batch = batch.max(dim=0)[0]

        mdp_psnr = calculate_mdp_psnr(mdp_output, mdp_batch)

        if not is_predict:
            self.log_step('val/mean_psnr', mean_psnr, on_step=False, on_epoch=True, prog_bar=True)
            self.log_step('val/mdp_psnr', mdp_psnr, on_step=False, on_epoch=True,)

        # visualization
        recon_images = view_recon(pred=mdp_output, gt=mdp_batch)[0]
        recon_images = Image.fromarray(recon_images)
        recon_images.save(os.path.join(self.save_folder, "recon", f"epoch_{self.global_step:05d}.png"))
        self.logger.experiment.log({"val_image": [wandb.Image(recon_images, caption="val_image")]}, step=self.global_step)
        
        view_on_image, view_on_image_post, view_on_image_cos, view_on_image_ref, view_classes = self.gs_model.visualize_points(
            xys=xys, 
            batch=mdp_batch,
            mdp_dapi_image=self.mdp_dapi_image,
            post_th=self.hparams['process']['bg_filter_th'],
            rna_class=self.rna_class, 
            rna_name=self.rna_name,
            selected_classes=self.hparams['view']['classes'],
        )

        view_on_image.save(os.path.join(self.save_folder, f"positions_mdp.png"))
        view_on_image_post.save(os.path.join(self.save_folder, f"positions_mdp_post.png"))
        view_on_image_cos.save(os.path.join(self.save_folder, f"positions_mdp_cos.png"))
        view_on_image_ref.save(os.path.join(self.save_folder, f"positions_mdp_ref.png"))
        
        for i, view_class in enumerate(view_classes):
            view_class.save(os.path.join(self.save_folder, 'classes', f"positions_class_{self.hparams['view']['classes'][i]}.png"))

        # visualization
        vmax = float(mdp_batch.data.ravel().max())
        vmin = float(mdp_batch.ravel()[mdp_batch.ravel() > 0].min()) if (mdp_batch.ravel() > 0).sum() > 0 else 0
        recon_images = [Image.fromarray(view_recon(pred=x, gt=y, vmax=vmax, vmin=vmin)[0]) for x, y in zip(output, batch)]

        for i, image in enumerate(recon_images):
            image.save(os.path.join(self.save_folder, "recon_plane", f"epoch_{self.global_step:05d}_{i}.png"))
        
        # 3d visualization
        self.gs_model.visualize_3d(save_folder=self.save_folder)

        if not is_predict:
            self.logger.experiment.log({
                "positions": [
                    wandb.Image(view_on_image, caption="mdp"),
                    wandb.Image(view_on_image_post, caption="mdp_post"),
                    wandb.Image(view_on_image_cos, caption="mdp_cos"),
                    wandb.Image(view_on_image_ref, caption="mdp_ref"),
                ],
                "positions_classes": [
                    wandb.Image(x, caption=f"pos_{self.hparams['view']['classes'][i]}") for i, x in enumerate(view_classes)
                ],
            })

            if log_each_plane:
                self.logger.experiment.log({
                    "recon_plane": [
                        wandb.Image(x, caption=f"plane_{i}") for i, x in enumerate(recon_images)
                    ],
                })

        if is_predict or (self.global_step > 0):
            self.gs_model.save_to_csv(
                xys=xys,
                batch=mdp_batch,
                rna_class=self.rna_class,
                rna_name=self.rna_name,
                hw=self.hparams['hw'],
                post_th=self.hparams['process']['bg_filter_th'],
                path=os.path.join(self.save_folder, "results.csv"),
            )

        if is_predict:
            # merge all images
            view_classes = [np.array(x) for x in view_classes]  # [(h, w, 3) * 8]
            view_classes = np.concatenate(view_classes, axis=1) # (h, 8w, 3)
            view_classes = Image.fromarray(view_classes)
            view_classes.save(os.path.join(self.save_folder, f"positions_classes.png"))

            # show top k classes
            top_classes, selected_classes = self.gs_model.visualize_top_classes(
                xys=xys, 
                batch=mdp_batch,
                mdp_dapi_image=self.mdp_dapi_image,
                post_th=self.hparams['process']['bg_filter_th'],
                rna_class=self.rna_class, 
                rna_name=self.rna_name,
                top_k=10,
            )

            os.makedirs(os.path.join(self.save_folder, 'classes_top10'), exist_ok=True)
            for i, selected_class in enumerate(top_classes):
                selected_class.save(os.path.join(self.save_folder, 'classes_top10', f"positions_{i}_{selected_classes[i]}.png"))
            
            # show score distribution
            self.gs_model.visualize_score_dist(
                xys=xys, 
                batch=mdp_batch,
                post_th=self.hparams['process']['bg_filter_th'],
                rna_class=self.rna_class, 
                rna_name=self.rna_name,
                save_folder=self.save_folder,
                selected_classes=self.hparams['view']['classes'],
            )
                
        return mean_psnr

    def log_step(self, name, loss, on_step=True, on_epoch=False, prog_bar=False):
        self.log(name, loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def validation_step(self, batch, batch_idx):
        batch, index = batch
        self.validation_step_outputs.append((self.obtain_output(index), batch))
    
    def on_validation_epoch_end(self):
        recon = [x[0][0] for x in self.validation_step_outputs]
        gt = [x[1] for x in self.validation_step_outputs]
        xys = self.validation_step_outputs[0][0][1] # (n, 2)

        self.validation_step_outputs.clear()

        recon = torch.cat(recon, dim=0) # (k, h, w, c)
        gt = torch.cat(gt, dim=0) # (k, h, w, c)

        self.forward_evaluation(recon, gt, xys, is_predict=False, log_each_plane=True)

    def predict_step(self, batch, batch_idx):
        batch, index = batch
        self.predict_step_outputs.append((self.obtain_output(index), batch))
    
    def on_predict_epoch_end(self):
        recon = [x[0][0] for x in self.predict_step_outputs]
        gt = [x[1] for x in self.predict_step_outputs]
        xys = self.predict_step_outputs[0][0][1] # (n, 2)

        self.predict_step_outputs.clear()

        recon = torch.cat(recon, dim=0) # (k, h, w, c)
        gt = torch.cat(gt, dim=0) # (k, h, w, c)

        self.forward_evaluation(recon, gt, xys, is_predict=True)

    def prune_points(self):
        # find indices to remove and update the persistent mask
        rgbs = torch.sigmoid(self.gs_model.rgbs)
        opacities = torch.sigmoid(self.gs_model.opacities)
        colors = rgbs * opacities

        indices_to_remove = (colors.max(dim=-1)[0] < self.hparams['train']['densification_th']).nonzero(as_tuple=True)[0]
        self.gs_model.maskout(indices_to_remove)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['gs_model'] = self.gs_model
        checkpoint['mdp_dapi_image'] = self.mdp_dapi_image

    def on_load_checkpoint(self, checkpoint):
        self.gs_model = checkpoint['gs_model']
        self.mdp_dapi_image = checkpoint['mdp_dapi_image']

    def _original(self, x):
        x = (x - self.hparams['train']['color_bias']) / (1 - self.hparams['train']['color_bias'] * 2)
        x = x * (self.hparams['value_range'][1] - self.hparams['value_range'][0]) + self.hparams['value_range'][0]

        return x
    
        