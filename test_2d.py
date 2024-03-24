import torch
import logging
logging.getLogger('lightning').setLevel(0)

from argparse import ArgumentParser
from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader

from systems_2d import GSSystem
from train_2d import RNADataset

torch.set_float32_matmul_precision('medium')
    

def main():
    parser = ArgumentParser()
    parser.add_argument("--devices", nargs='+', default=[0])
    parser.add_argument("--checkpoint_path", type=str, default='outputs/c6oci7nv/checkpoints/last.ckpt')
    args = parser.parse_args()

    # fix seed
    seed_everything(42)

    # model & dataloader
    gs_system = GSSystem.load_from_checkpoint(args.checkpoint_path)
    gs_system.hparams['view'] = {'classes': ['Snap25', 'Slc17a7', 'Gad1', 'Gad2', 'plp1', 'MBP', 'Aqp4', 'Rgs5']}

    val_dataset = RNADataset(hparams=gs_system.hparams, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=27)

    trainer = Trainer(enable_model_summary=False)

    trainer.predict(model=gs_system, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()