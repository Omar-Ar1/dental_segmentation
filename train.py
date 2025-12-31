import os

# --- CRITICAL: Set these BEFORE importing torch/lightning ---
# Redirect all temporary and cache files to Workdir to avoid Home Inode limits
WORK_DIR = "/gpfs/workdir/arbiom"
os.environ["TMPDIR"] = os.path.join(WORK_DIR, "tmp")
os.environ["TORCH_HOME"] = os.path.join(WORK_DIR, ".cache/torch")
os.environ["HF_HOME"] = os.path.join(WORK_DIR, ".cache/huggingface")
os.environ["MPLCONFIGDIR"] = os.path.join(WORK_DIR, ".cache/matplotlib")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.module import DentalSegmentationModule
from src.data.datamodule import DentalDataModule

# MONAI transforms (your code was mixing torchvision syntax)
from monai.transforms import (
    Compose,
    ResizeD,
    RandFlipD,
    ToTensorD,
)
import torch



def main():
    # Configuration (Could be loaded from a YAML file via Hydra)
    config = {
        "json_path": "dataset/annotations.json",
        "img_dir": "dataset/images",
        "backbone": "vit_large_patch16_dinov3.lvd1689m",
        "batch_size": 8,
        "lr": 1e-3,
        "epochs": 100,
        "accum": 4,
        "seed": 42,
    }

    L.seed_everything(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    # 1. Init Data
    dm = DentalDataModule(
        json_path=config["json_path"],
        img_dir=config["img_dir"],
        batch_size=config["batch_size"],
    )

    # 2. Init Model
    model = DentalSegmentationModule(
        backbone_name=config["backbone"],
        num_classes=2,
        lr=config["lr"],
        num_register_tokens=4,
    )

    # 3. Callbacks
    checkpoints_dir = "/gpfs/workdir/arbiom/dental_segmentation/checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoints_dir,
        monitor="val/dice",
        mode="max",
        filename="best-{epoch:02d}-{val/dice:.4f}",
        save_last=True,
        save_weights_only=True,
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/dice",
        patience=10,
        mode="max",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Trainer
    trainer = L.Trainer(
        default_root_dir="/gpfs/workdir/arbiom/dental_segmentation/experiments",
        max_epochs=config["epochs"],
        accelerator='gpu',
        devices=torch.cuda.device_count(),
        strategy='ddp_find_unused_parameters_false',
        precision='bf16-mixed',
        accumulate_grad_batches=config["accum"],
        logger=TensorBoardLogger("/gpfs/workdir/arbiom/dental_segmentation/logs/", name="dino_seg"),
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps=10,
    )
    # 7. Resume Logic
    ckpt_path = None
    last_ckpt = os.path.join(checkpoints_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"🔄 Resuming training from: {last_ckpt}")
        ckpt_path = last_ckpt
    else:
        print("🚀 Starting new training run")

    # 5. Fit
    trainer.fit(model, dm, ckpt_path=ckpt_path)

    # 6. Test
    trainer.test(model, dm, ckpt_path="last")


if __name__ == "__main__":
    main()
