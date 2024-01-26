import torch
import tyro
import wandb
import os

from pathlib import Path
from typing import Optional

from systems import SimpleTrainer
from preprocess import images_to_tensor

def main(
    primary_samples: int = 8000,
    backup_samples: int = 8000,
    save_imgs: bool = True,
    img_path: Optional[Path] = Path("data/1213_demo_data_v2/raw1"),
    codebook_path: Optional[Path] = Path("data/codebook.xlsx"),
    iterations: int = 20000,
    densification_interval: int = 1000,
    lr: float = 0.002,
    exp_name: str = "debug",
    cali_loss_type: str = "cos",
    initialization: bool = False,
    pos_score: int = 1,
    eval:bool = False,
    model_path:str = "RNA_GS/outputs/debug/params.pth",
    weights: list[float] = [
        0,
        1,
        0,
        0,
        0,
        0,
        0.001,
        0.1,
        0,
        0,
        0,
    ],  # l1, l2, lml1, lml2, bg, ssim, code_cos, circle, size, rho, mdp(maximum density projection), mip(mean intensity projection)
    thresholds: list[float] = [
        0.1,
        0.04,
        0.04,
    ],  # prune, grad, gauss th
    dens_flags: list[int] = [
        False,
        False,
        False
    ], # prune, split, clone
    size_range : list[int] = [1, 3],
) -> None:
    
    config = {
        "w_l1": weights[0],
        "w_l2": weights[1],
        "w_lml1": weights[2],
        "w_lml2": weights[3],
        "w_bg": weights[4],
        "w_ssim": weights[5],
        "w_code": weights[6],
        "w_circle": weights[7],
        "w_size": weights[8],
        "w_rho": weights[9],
        "w_mdp": weights[10],
        "w_mip": weights[11],
        "prune_threshold" : thresholds[0],
        "grad_threshold" : thresholds[1],
        "gauss_threshold" : thresholds[2],
        "prune_flag" : dens_flags[0],
        "split_flag" : dens_flags[1],
        "clone_flag" : dens_flags[2],        
        "exp_name": exp_name,
        "codebook_path": codebook_path,
        "cali_loss_type": cali_loss_type,
        "primary_samples": primary_samples,
        "backup_samples": backup_samples,
        "densification_interval": densification_interval,
        "initialization": initialization,
        "pos_score": pos_score,
        "size_range": size_range,
    }

    wandb.init(
        project="rna_cali_0120",
        config=config,
        name=config["exp_name"],
    )

    print(f"Running with config: {config}")

    gt_image = images_to_tensor(img_path)

    trainer = SimpleTrainer(
        gt_image=gt_image,
        primary_samples=primary_samples,
        backup_samples=backup_samples,
        cfg=config,
        image_file_name=img_path,
        densification_interval=densification_interval,
    )
    if eval:
        os.environ['WANDB_MODE'] = 'offline'
        trainer.test(model_path=model_path)        
    else:
        trainer.train(
            iterations=iterations,
            lr=lr,
            save_imgs=save_imgs,
        )


if __name__ == "__main__":
    tyro.cli(main)