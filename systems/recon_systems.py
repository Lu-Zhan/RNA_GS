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


class GSSystem3D(LightningModule):
    def __init__(self, hparams, **kwargs):  
        super().__init__()
        self.save_hyperparameters(hparams)

        # save folder
        self.save_folder = hparams['exp_dir']
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "recon"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, "recon_plane"), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, 'classes'), exist_ok=True)
        os.makedirs(os.path.join(self.save_folder, 'view_3d'), exist_ok=True)

        self.gt_2d = None
        self.gt_3d = None
        self.is_init_rgb = False

        self.cam_model = SliceCameras(
            num_cams=len(hparams['camera']['cam_ids']),
            num_slices=hparams['camera']['num_slices'],
            num_dims=self.hparams['model']['num_dims'],
            hw=self.hparams['hw'],
            step_z=self.hparams['model']['step_z'],
            camera_z=self.hparams['model']['camera_z'],
            refine_camera=self.hparams['train']['refine_camera'],
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
            num_dims = self.hparams['model']['num_dims']
            self.rna_class, self.rna_name = np.array([[1.] * num_dims]), np.array(['full'])
            self.mi_rna_class, self.mi_rna_name = np.array([[1.] * num_dims, [0.] * num_dims]), np.array(['full', 'background'])

        self.rna_class = torch.tensor(self.rna_class, device=self.gs_model.means_3d.device, dtype=self.gs_model.means_3d.dtype)
        self.mi_rna_class = torch.tensor(self.mi_rna_class, device=self.gs_model.means_3d.device, dtype=self.gs_model.means_3d.dtype)

        self.dapi_images = kwargs.get('dapi_images', None)
        try:
            self.mdp_dapi_image = self.dapi_images.max(dim=-1)[0].max(dim=0)[0]
        except:
            print("dapi_images not available, read from checkpoint")

        self.validation_step_outputs = []
        self.predict_step_outputs = []

        if self.hparams['train']['refine_camera']:
            self.automatic_optimization = False

    def configure_optimizers(self):
        gs_optimizer = optim.Adam(self.gs_model.parameters, lr=self.hparams['train']['lr'])

        if self.hparams['train']['refine_camera']:
            cam_optimizer = optim.Adam(self.cam_model.parameters, lr=self.hparams['train']['lr_cam'])
            return [gs_optimizer, cam_optimizer]
        else:
            return [gs_optimizer]

    def training_step(self, batch, batch_idx):
        gt_3d, cam_indexs, slice_indexs = batch

        if self.is_init_rgb is False and self.hparams['train']['init_rgb']:
            with torch.no_grad():
                xys = self.gs_model.obtain_xys(camera=self.cam_model)
                self.gs_model.init_rgbs(xys=xys, gt_images=gt_3d.max(dim=0)[0])
            self.is_init_rgb = True

        gt_2d = gt_3d.max(dim=0)[0]    # (k, h, w, c) -> (h, w, c)
        gt_mdp = gt_2d.max(dim=-1)[0]  # (h, w, c) -> (h, w)

        den_interval = self.hparams['train']['densification_interval']
        den_start = self.hparams['train']['densification_start']

        if den_interval > 0 and self.hparams['train']['model'] == 'guass':
            if self.global_step % (den_interval + 1) == 0 and self.global_step > den_start:
                self.prune_points()

        recon_3d, cov2d, radii, _ = self.gs_model.render_slice(
            camera=self.cam_model, cam_indexs=cam_indexs, slice_indexs=slice_indexs,
        )

        recon_2d = recon_3d.max(dim=0)[0]  #.max(dim=0)[0]   # (n, h, w, k) -> (h, w, k)
        recon_mdp = recon_2d.max(dim=-1)[0]  #.max(dim=0)[0]   # (h, w, k) -> (h, w)
        
        results = {
            '3d': (recon_3d, gt_3d),
            '2d': (recon_2d, gt_2d),
            'mdp': (recon_mdp, gt_mdp),
            'cov2d': cov2d,
            'radii': radii,
        }

        loss = self.compute_loss(results)
        
        self.log_step("train/total_loss", loss, prog_bar=True)
        self.log_step("params/num_samples", self.gs_model.current_num_samples)
        self.logger.experiment.log({
            f"cam_zs/{self.hparams['camera']['cam_ids'][i]}": self.cam_model.camera_zs[i] for i in self.hparams['camera']['cam_indexs']
        })
        self.log_step("params/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        if self.hparams['train']['refine_camera']:
            self.log_step("params/lr_cam", self.trainer.optimizers[1].param_groups[0]['lr'])

        if self.hparams['train']['refine_camera']:
            opt_gs, opt_cam = self.optimizers()
            opt_gs.zero_grad()
            opt_cam.zero_grad()
            self.manual_backward(loss)
            opt_gs.step()
            opt_cam.step()

        return loss

    def compute_loss(self, results):
        loss = 0.

        for recon_type in ['3d', '2d', 'mdp']:
            pred_data, gt_data = results[recon_type]
            if self.hparams['loss'][recon_type]['w_l1'] > 0:
                loss_l1 = l1_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_l1'] * loss_l1
                self.log_step(f"train_{recon_type}/loss_l1", loss_l1)
            
            if self.hparams['loss'][recon_type]['w_masked_l1'] > 0:
                loss_masked_l1 = masked_l1_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_masked_l1'] * loss_masked_l1
                self.log_step(f"train_{recon_type}/loss_masked_l1", loss_masked_l1)
            
            if self.hparams['loss'][recon_type]['w_bg_l1'] > 0:
                loss_bg_l1 = bg_l1_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_bg_l1'] * loss_bg_l1
                self.log_step(f"train_{recon_type}/loss_bg_l1", loss_bg_l1)
            
            if self.hparams['loss'][recon_type]['w_l2'] > 0:
                loss_l2 = mse_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_l2'] * loss_l2
                self.log_step(f"train_{recon_type}/loss_l2", loss_l2)
            
            if self.hparams['loss'][recon_type]['w_masked_l2'] > 0:
                loss_masked_l2 = masked_mse_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_masked_l2'] * loss_masked_l2
                self.log_step(f"train_{recon_type}/loss_masked_l2", loss_masked_l2)
            
            if self.hparams['loss'][recon_type]['w_bg_l2'] > 0:
                loss_bg_l2 = bg_mse_loss(pred_data, gt_data)
                loss += self.hparams['loss'][recon_type]['w_bg_l2'] * loss_bg_l2
                self.log_step(f"train_{recon_type}/loss_bg_l2", loss_bg_l2)
            
        if self.hparams['loss']['shape']['w_rho'] > 0:
            loss_rho = rho_loss(results['cov2d'])
            loss += self.hparams['loss']['w_rho'] * loss_rho
            self.log_step("train/loss_rho", loss_rho)
        
        if self.hparams['loss']['shape']['w_radius'] > 0:
            loss_radius = radius_loss(results['radii'].to(self.gs_model.means_3d.dtype))
            loss += self.hparams['loss']['w_radius'] * loss_radius
            self.log_step("train/loss_radius", loss_radius)
        
        # pred_code = self.gs_model.colors
        # loss_mi = mi_loss(pred_code, self.mi_rna_class)
        # self.log_step("train/loss_mi", loss_mi)

        # loss_cos = cos_loss(pred_code, self.rna_class)
        # self.log_step("train/loss_cos", loss_cos)

        # if self.hparams['loss']['class']['w_mi'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
        #     loss += self.hparams['loss']['w_mi'] * loss_mi

        # if self.hparams['loss']['class']['w_cos'] > 0 and self.global_step > self.hparams['train']['codebook_start']:
        #     loss += self.hparams['loss']['w_cos'] * loss_cos
        return loss
    
    def obtain_output(self, cam_indexs, slice_indexs):
        output, _, _, xys = self.gs_model.render_slice(
            camera=self.cam_model, cam_indexs=cam_indexs, slice_indexs=slice_indexs,
        )

        return output, xys

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
        
        view_on_image, view_on_image_post, view_on_image_cos, view_on_image_ref, view_classes = self.gs_model.visualize_points(
            xys=xys, 
            batch=gt_2d,
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
        vmax = float(gt_2d.data.ravel().max())
        vmin = float(gt_2d.ravel()[gt_2d.ravel() > 0].min()) if (gt_2d.ravel() > 0).sum() > 0 else 0
        recon_images = [Image.fromarray(view_recon(pred=x, gt=y, vmax=vmax, vmin=vmin)[0]) for x, y in zip(recon_3d, gt_3d)]

        for i, image in enumerate(recon_images):
            image.save(os.path.join(self.save_folder, "recon_plane", f"iter_{self.global_step:05d}_{i}.png"))
        
        # 3d visualization
        self.gs_model.visualize_3d(save_path=os.path.join(self.save_folder, "view_3d", f"iter_{self.global_step:05d}.ply"))

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
                batch=gt_2d,
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
                batch=gt_2d,
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
                batch=gt_2d,
                post_th=self.hparams['process']['bg_filter_th'],
                rna_class=self.rna_class, 
                rna_name=self.rna_name,
                save_folder=self.save_folder,
                selected_classes=self.hparams['view']['classes'],
            )
                
        return mean_3d_psnr

    def log_step(self, name, loss, on_step=True, on_epoch=False, prog_bar=False):
        self.log(name, loss, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def validation_step(self, batch, batch_idx):
        batch, cam_indexs, slice_indexs = batch
        self.validation_step_outputs.append((self.obtain_output(cam_indexs, slice_indexs), batch))
    
    def on_validation_epoch_end(self):
        recon = [x[0][0] for x in self.validation_step_outputs]
        gt = [x[1] for x in self.validation_step_outputs]
        xys = self.validation_step_outputs[0][0][1] # (n, 2)

        self.validation_step_outputs.clear()

        recon = torch.cat(recon, dim=0) # (k, h, w, c)
        gt = torch.cat(gt, dim=0) # (k, h, w, c)

        self.forward_evaluation(recon, gt, xys, is_predict=False, log_each_plane=True)

    def predict_step(self, batch, batch_idx):
        batch, cam_indexs, slice_indexs = batch
        self.predict_step_outputs.append((self.obtain_output(cam_indexs, slice_indexs), batch))

        return
    
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
    
        