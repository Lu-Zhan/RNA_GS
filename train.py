import os
import yaml
import torch
import logging
logging.getLogger('lightning').setLevel(0)

from argparse import ArgumentParser, REMAINDER
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from systems import GSSystem3D
from datasets import RNADataset3DRand

torch.set_float32_matmul_precision('medium')

def main():
    parser = ArgumentParser()
    parser.add_argument("--devices", nargs='+', default=[0])
    parser.add_argument("--config", type=str, default='configs/crop64_rawtiff.yaml')
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--exp_dir", type=str, default='outputs_3d_rand/')
    parser.add_argument("extra", nargs=REMAINDER, help='Modify hparams.')
    args = parser.parse_args()
    
    config = obtain_config(args)

    # fix seed
    seed_everything(42)

    # logging
    logger = WandbLogger(
        name=config['exp_name'],
        offline=False,
        project='rna_cali_0320',
        save_dir=config['exp_dir'],
    )

    ckpt_path = os.path.join(logger.save_dir, logger.experiment.id, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    config['exp_dir'] = os.path.dirname(ckpt_path)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename='psnr={val/mean_psnr:.4f}',
        save_last=True,
        save_top_k=1,
        monitor='val/mean_psnr',
        auto_insert_metric_name=False,
        mode='max',
    )

    # model & dataloader
    train_dataset = RNADataset3DRand(hparams=config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True, num_workers=8)

    val_dataset = RNADataset3DRand(hparams=config, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    config['hw'] = train_dataset.size[:2]
    config['value_range'] = train_dataset.range
    gs_system = GSSystem3D(hparams=config, dapi_images=train_dataset.dapi_images)

    trainer = Trainer(
        benchmark=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=config['devices'],
        accelerator="gpu",
        max_epochs=config['train']['max_epochs'],
        precision="32-true",
        log_every_n_steps=50,
        strategy="auto",
        check_val_every_n_epoch=2,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )

    trainer.fit(
        model=gs_system, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    gs_system = GSSystem3D.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.predict(model=gs_system, dataloaders=val_dataloader)


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