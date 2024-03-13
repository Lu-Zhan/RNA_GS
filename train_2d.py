import os
import yaml
import torch
import logging
logging.getLogger('lightning').setLevel(0)

from pathlib import Path
from argparse import ArgumentParser
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

from systems_2d import GSSystem
from preprocess import images_to_tensor, images_to_tensor_cropped

torch.set_float32_matmul_precision('medium')

class RNADataset(Dataset):
    def __init__(self, hparams, mode='train'):  
        path = Path(hparams['data']['data_path'])
        try:
            self.gt_image = images_to_tensor(path)
        except:
            self.gt_image = images_to_tensor_cropped(path)
        
        self.num_iters = hparams['train']['iterations']
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return self.num_iters
        else:
            return 1

    def __getitem__(self, index):
        return self.gt_image

    @property
    def size(self):
        return self.gt_image.shape
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--devices", nargs='+', default=[0])
    parser.add_argument("--config", type=str, default='configs/default.yaml')
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--exp_dir", type=str, default='outputs/')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    config['devices'] = args.devices
    config['config'] = args.config
    config['exp_dir'] = args.exp_dir

    if args.exp_name != '':
        config['exp_name'] = args.exp_name

    # fix seed
    seed_everything(42)

    # logging
    logger = WandbLogger(
        name=config['exp_name'],
        offline=False,
        project='rna_cali_0311',
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
    train_dataset = RNADataset(hparams=config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=27)

    val_dataset = RNADataset(hparams=config, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=27)

    config['hw'] = train_dataset.size[:2]
    gs_system = GSSystem(hparams=config)

    trainer = Trainer(
        benchmark=True,
        logger=logger,
        callbacks=[checkpoint_callback],
        devices=config['devices'],
        accelerator="gpu",
        max_epochs=-1,
        max_steps=config['train']['iterations'],
        precision="32-true",
        log_every_n_steps=50,
        strategy="auto",
        val_check_interval=1000,
        enable_model_summary=False,
        num_sanity_val_steps=1,
    )

    trainer.fit(
        model=gs_system, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.predict(model=gs_system, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()