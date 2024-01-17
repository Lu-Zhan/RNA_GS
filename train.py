import torch
import tyro
import wandb

from pathlib import Path
from typing import Optional

from systems import SimpleTrainer
from preprocess import images_to_tensor

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 8000,
    save_imgs: bool = True,
    img_path: Optional[Path] = Path("data/1213_demo_data_v2/raw1"),
    codebook_path: Optional[Path] = Path("data/codebook.xlsx"),
    iterations: int = 20000,
    densification_interval: int = 1000,
    lr: float = 0.002,
    exp_name: str = "debug",
    cali_loss_type: str = "cos",
    weights: list[float] = [
        0,
        1,
        0,
        0,
        0,
        0,
        0.001,
        0.1,
    ],  # l1, l2, lml1, lml2, bg, ssim, code_cos, scale
    thresholds: list[float] = [
        0.3,
        0.04,
        0.04,
    ],  # l1, l2, lml1, lml2, bg, ssim, code_cos
) -> None:
    config = {
        "w_l1": weights[0],
        "w_l2": weights[1],
        "w_lml1": weights[2],
        "w_lml2": weights[3],
        "w_bg": weights[4],
        "w_ssim": weights[5],
        "w_code_cos": weights[6],
        "w_scale": weights[7],
        "prune_threshold" : thresholds[0],
        "grad_threshold" : thresholds[1],
        "gauss_threshold" : thresholds[2],
        "exp_name": exp_name,
        "codebook_path": codebook_path,
        "cali_loss_type": cali_loss_type,
    }

    wandb.init(
        project="rna_cali",
        config=config,
        name=config["exp_name"],
    )

    print(f"Running with config: {config}")

    if img_path:
        # gt_image = image_path_to_tensor(img_path)
        gt_image = images_to_tensor(img_path)

    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(
        gt_image=gt_image,
        num_points=num_points,
        cfg=config,
        image_file_name=img_path,
        densification_interval=densification_interval,
    )
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
