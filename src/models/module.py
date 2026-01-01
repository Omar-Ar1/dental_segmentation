import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from src.models.components.dinov2 import DinoV2Segmentation
from src.utils.metrics import dice_score, Hausdorff_dist

class DentalSegmentationModule(L.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        num_register_tokens: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        taps: list = [12, 18, 23],
        use_boundary_loss: bool = True,
        lambda_b: float = 0.2,
        feat_channels: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = DinoV2Segmentation(
            backbone_name=backbone_name,
            num_classes=num_classes,
            taps=taps,
            num_register_tokens=num_register_tokens,
            feat_channels=feat_channels,
        )
        
        self.seg_criterion = nn.CrossEntropyLoss()
        self.bnd_criterion = nn.BCEWithLogitsLoss() if use_boundary_loss else None

    def forward(self, x):
        return self.model(x)

    def _make_boundary_target(self, mask):
        # Create boundary target on the fly
        m = (mask > 0).float().unsqueeze(1)
        eroded = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
        return (m - eroded).clamp_min(0)

    def _shared_step(self, batch, stage="train"):
        img, mask = batch
        logits, boundary_logits = self(img)
        
        # Segmentation Loss
        loss_seg = self.seg_criterion(logits, mask)
        
        # Boundary Loss (Optional)
        loss_bnd = 0.0
        if self.hparams.use_boundary_loss and boundary_logits is not None:
            bnd_target = self._make_boundary_target(mask)
            loss_bnd = self.bnd_criterion(boundary_logits, bnd_target)
            
        total_loss = loss_seg + (self.hparams.lambda_b * loss_bnd)

        # Metrics
        with torch.no_grad():
            dice = dice_score(logits, mask)
            # Only compute expensive HD in validation/test
            hd = 0.0
            if stage != "train":
                hd = Hausdorff_dist(logits, mask)

        self.log(f"{stage}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/dice", dice, on_epoch=True, prog_bar=True)
        if stage != "train":
            self.log(f"{stage}/hd", hd, on_epoch=True)
            
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]