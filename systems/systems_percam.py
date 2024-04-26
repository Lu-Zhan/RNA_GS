from systems.systems import *


class GSSystem3DPerCam(GSSystem3D):
    def training_step(self, batch, batch_idx):
        batch, cam_idxs, slice_idxs = [x[0] for x in batch]

        if self.is_init_rgb is False and self.hparams['train']['init_rgb']:
            with torch.no_grad():
                xys = self.gs_model.obtain_xys(camera=self.cam_model)
                self.gs_model.init_rgbs(xys=xys, gt_images=batch.max(dim=0)[0])
            self.is_init_rgb = True

        if self.mdp_image is None:
            self.mdp_image = batch.max(dim=-1)[0]   #.max(dim=0)[0]

        den_interval = self.hparams['train']['densification_interval']
        den_start = self.hparams['train']['densification_start']

        if den_interval > 0 and self.hparams['train']['model'] == 'guass':
            if self.global_step % (den_interval + 1) == 0 and self.global_step > den_start:
                self.prune_points()

        output, cov2d, radii, _ = self.gs_model.render_slice(
            camera=self.cam_model, cam_idxs=cam_idxs, slice_idxs=slice_idxs,
        )

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

        if self.hparams['loss']['w_rho'] > 0:
            loss_rho = rho_loss(cov2d)
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
        self.log_step("params/num_samples", self.gs_model.current_num_samples)
        self.logger.experiment.log({
            f"cam_zs/{i}": self.cam_model.camera_zs[i] for i in self.hparams['camera']['cam_ids']
        })
        self.log_step("params/lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        if self.hparams['train']['refine_camera']:
            self.log_step("params/lr_cam", self.trainer.optimizers[1].param_groups[0]['lr'])

        if self.hparams['train']['refine_camera']:
            opt_gs, opt_cam = self.optimizers()
            opt_gs.zero_grad()

            if self.current_epoch > self.hparams['train']['refine_camera_start_epoch']:
                opt_cam.zero_grad()

            self.manual_backward(loss)
            opt_gs.step()

            if self.current_epoch > self.hparams['train']['refine_camera_start_epoch']:
                opt_cam.step()

        return loss
        output = self._original(output)
        input = self._original(input)

        # metric
        mean_mse = torch.mean((output - input) ** 2).cpu()
        mean_psnr = float(10 * torch.log10(1 / mean_mse))

        mdp_output = output.max(dim=0)[0]
        mdp_batch = input.max(dim=0)[0]

        mdp_psnr = calculate_mdp_psnr(mdp_output, mdp_batch)

        if not is_predict:
            self.log_step('val/mean_psnr', mean_psnr, on_step=False, on_epoch=True, prog_bar=True)
            self.log_step('val/mdp_psnr', mdp_psnr, on_step=False, on_epoch=True,)

        # visualization
        recon_images = view_recon(pred=mdp_output, gt=mdp_batch)[0]
        recon_images = Image.fromarray(recon_images)
        recon_images.save(os.path.join(self.save_folder, "recon", f"iter_{self.global_step:05d}.png"))
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
        recon_images = [Image.fromarray(view_recon(pred=x, gt=y, vmax=vmax, vmin=vmin)[0]) for x, y in zip(output, input)]

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