import os
import yaml
import torch
import logging
import wandb
import numpy as np
logging.getLogger('lightning').setLevel(0)

from pathlib import Path
from argparse import ArgumentParser, REMAINDER
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from visualize import view_recon, view_positions
from utils import read_codebook, filter_by_background

from systems_2d import GSSystem
from datasets import RNADataset

torch.set_float32_matmul_precision('medium')


def main():
    parser = ArgumentParser()
    parser.add_argument("--devices", nargs='+', default=[0])
    parser.add_argument("--config", type=str, default='configs/default.yaml')
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--exp_dir", type=str, default='outputs/')
    # parser.add_argument("--classes", nargs='+', type=str, default='Snap25')
    # classes = Snap25 Slc17a7 Gad1 Gad2 plp1 MBP GFAP Aqp4 Rgs5
    parser.add_argument("extra", nargs=REMAINDER, help='Modify hparams.')
    args = parser.parse_args()
    
    config = obtain_config(args)

    # fix seed
    seed_everything(42)

    # logging
    logger = WandbLogger(
        name=config['exp_name'],
        offline=False,
        project='rna_cali_0311',
        save_dir=config['exp_dir'],
    )

    ckpt_path = os.path.join(logger.save_dir, config['exp_name'], 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    config['exp_dir'] = os.path.dirname(ckpt_path)
    device_list = [torch.device(f"cuda:{d}") for d in args.devices]

    # model & dataloader
    train_dataset = RNADataset(hparams=config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)

    val_dataset = RNADataset(hparams=config, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    config['hw'] = train_dataset.size[:2]
    config['value_range'] = train_dataset.range
    gs_system = GSSystem(hparams=config, dapi_images=train_dataset.dapi_images)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='psnr={val/mean_psnr:.4f}',
        save_last=True,
        save_top_k=1,
        monitor='val/mean_psnr',
        auto_insert_metric_name=False,
        mode='max',
    )
    trainer = Trainer(
        logger=logger
    )

    path = os.path.join(ckpt_path, "last.ckpt")
    gs_system = GSSystem.load_from_checkpoint(path)
    with torch.no_grad():
        classes_list = ["Snap25", "Slc17a7", "Gad1", "Gad2", "plp1", "MBP", "GFAP", "Aqp4", "Rgs5"]
        for i in classes_list:
            _, _, _, xys = gs_system.forward()
            cos_score, pred_rna_index, pred_rna_name = gs_system.gs_model.obtain_calibration(
                gs_system.rna_class,
                gs_system.rna_name)
            ind = pred_rna_name == i
            print("number:",np.sum(pred_rna_name == i))
            if np.sum(pred_rna_name == i) == 0:
                continue
            # print(train_dataset.gt_images.shape)
            
            # visualize_points
            xys=xys[ind]
            batch=train_dataset.gt_images.to(device=device_list[0])
            mdp_dapi_image=gs_system.mdp_dapi_image
            post_th=gs_system.hparams['process']['bg_filter_th']
            rna_class=gs_system.rna_class
            rna_name=gs_system.rna_name
            points_xy = xys.cpu().numpy()
            mdp_dapi_image = mdp_dapi_image.cpu().numpy()
            mdp_image = batch.max(dim=-1)[0].cpu().numpy()

            max_color = gs_system.gs_model.colors[ind].max(dim=-1)[0]
            max_color = max_color / (max_color.max() + 1e-8)
            th=post_th
            max_color_post = filter_by_background(
                xys=xys,
                colors=gs_system.gs_model.colors[ind],
                hw=[gs_system.gs_model.H, gs_system.gs_model.W],
                image=batch,
                th=th,
            )

            max_color_post = max_color_post.max(dim=-1)[0]
            max_color_post = max_color_post / (max_color_post.max() + 1e-8)

            cos_score = gs_system.gs_model.obtain_calibration(rna_class, rna_name)[0]

            ref_score = cos_score[ind] * max_color_post
            
            view_on_image = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color.cpu().numpy())
            view_on_image_post = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=max_color_post.cpu().numpy())
            view_on_image_ref = view_positions(points_xy=points_xy, bg_image=mdp_image, alpha=ref_score.cpu().numpy())
            logger.experiment.log({"positions": [
                    wandb.Image(view_on_image, caption=f"{i}"),
                ]})
            view_on_image_post.save(os.path.join(gs_system.save_folder, f"{i}_post.png"))
            # view_on_image_ref.save(os.path.join(gs_system.save_folder, f"{args.classes[0]}_ref.png"))

         


def obtain_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config['devices'] = args.devices
    config['config'] = args.config
    config['exp_dir'] = args.exp_dir

    keys = args.extra[::2]
    values = args.extra[1::2]

    for k, v in zip(keys, values):
        kk = k.split('.')

        try:
            v = float(v)
        except:
            pass

        if len(kk) == 2:
            config[kk[0]][kk[1]] = v
        else:
            config[kk[0]] = v
        
    if args.exp_name != '':
        config['exp_name'] = args.exp_name

    return config


if __name__ == "__main__":
    main()