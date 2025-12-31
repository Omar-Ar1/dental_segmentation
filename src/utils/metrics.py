# stage1_segmentation/utils/metrics.py
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from monai.metrics import HausdorffDistanceMetric

def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    ignore_background: bool = True,
    num_classes: int | None = None,
    threshold: float = 0.5,   # used if C==1 probabilities; if logits, 0.0 is fine
) -> torch.Tensor:
    """
    Computes mean (macro) Dice. Supports:
      pred: [B,C,H,W] logits/probs  OR  [B,H,W] label/prob mask
      target: [B,H,W] integer labels (multi-class) or {0,1} for binary
    """
    device = pred.device

    # --- Normalize to one-hot [B,C,H,W] for pred and target ---
    if pred.ndim == 4:
        B, C, H, W = pred.shape

        if C == 1:
            # Treat as binary probabilities/logits -> 2-class labels
            # If these are logits, threshold=0.0 is typical; for probs, 0.5.
            pred_labels = (pred.squeeze(1) > threshold).long()
            C_eff = 2
        else:
            pred_labels = torch.argmax(pred, dim=1)  # [B,H,W]
            C_eff = C

        if target.dtype.is_floating_point:
            tgt_labels = (target > 0.5).long()
            C_tgt = 2
        else:
            C_tgt = int(target.max().item()) + 1 if target.numel() else 1
            tgt_labels = target.long()

        C_final = num_classes if num_classes is not None else max(C_eff, C_tgt)
        pred_1h = F.one_hot(pred_labels.clamp_min(0), num_classes=C_final).permute(0,3,1,2).float()
        tgt_1h  = F.one_hot(tgt_labels.clamp_min(0),  num_classes=C_final).permute(0,3,1,2).float()

    elif pred.ndim == 3:
        # Label or prob masks
        if pred.dtype.is_floating_point:
            pred_labels = (pred > 0.5).long()
            C_pred = 2
        else:
            pred_labels = pred.long()
            C_pred = int(pred_labels.max().item()) + 1 if pred.numel() else 1

        if target.dtype.is_floating_point:
            tgt_labels = (target > 0.5).long()
            C_tgt = 2
        else:
            tgt_labels = target.long()
            C_tgt = int(tgt_labels.max().item()) + 1 if target.numel() else 1

        C_final = num_classes if num_classes is not None else max(C_pred, C_tgt)
        pred_1h = F.one_hot(pred_labels.clamp_min(0), num_classes=C_final).permute(0,3,1,2).float()
        tgt_1h  = F.one_hot(tgt_labels.clamp_min(0),  num_classes=C_final).permute(0,3,1,2).float()
    else:
        raise ValueError(f"Unsupported pred shape {tuple(pred.shape)}")

    pred_1h = pred_1h.to(device)
    tgt_1h  = tgt_1h.to(device)

    # --- Dice per class ---
    dims = (0, 2, 3)
    intersection = torch.sum(pred_1h * tgt_1h, dims)
    cardinality  = torch.sum(pred_1h, dims) + torch.sum(tgt_1h, dims)
    dice_per_class = (2.0 * intersection + epsilon) / (cardinality + epsilon)  # [C]

    # Optionally ignore background (class 0), but only if present
    if ignore_background and dice_per_class.numel() > 1:
        # mask to classes present in target (avoid averaging empty / all-zero classes)
        present = (tgt_1h.sum(dim=dims) > 0)  # [C] bool
        present[0] = False  # drop background
        if present.any():
            return dice_per_class[present].mean()
        else:
            # no foreground present; fall back to background dice (or 1.0 if both empty)
            return dice_per_class[0]
    else:
        return dice_per_class.mean()

# utils/metrics_extra.py
# utils/metrics.py

def Hausdorff_dist(
    pred: torch.Tensor,
    target: torch.Tensor,
    percentile: int = 95,
    ignore_background: bool = True,
    threshold: float = 0.5,               # used when C==1 probs/logits
    num_classes: Optional[int] = None,    # force class count if you know it
    spacing: Optional[Tuple[float, ...]] = None,  # (sy,sx) or (sz,sy,sx)
    directed: bool = False,
) -> Dict[str, float]:
    """
    HD@percentile per class using MONAI, robust to binary/multiclass, 2D/3D, and empty classes.
    pred:   [B,C,*spatial] logits/probs OR [B,*spatial] label/prob mask
    target: [B,*spatial]   integer labels (multi) or {0,1} (binary)
    Returns dict with 'mean' and 'HD{percentile}_class_k' for k in [0..C-1].
    """
    device = pred.device

    # ---- normalize to one-hot [B,C,*spatial] ----
    if pred.ndim >= 4:  # [B,C,*]
        B, C = pred.shape[:2]
        if C == 1:
            labels = (pred.squeeze(1) > threshold).long()
            C_eff = 2
        else:
            labels = torch.argmax(pred, dim=1)
            C_eff = C
        if target.dtype.is_floating_point:
            tgt_labels = (target > 0.5).long()
            C_tgt = 2
        else:
            C_tgt = int(target.max().item()) + 1 if target.numel() else 1
            tgt_labels = target.long()
        C_final = num_classes if num_classes is not None else max(C_eff, C_tgt)
        y_pred = F.one_hot(labels.clamp_min(0), num_classes=C_final).movedim(-1, 1).float()
        y_true = F.one_hot(tgt_labels.clamp_min(0),  num_classes=C_final).movedim(-1, 1).float()

    elif pred.ndim == target.ndim and pred.ndim >= 3:  # [B,*]
        if pred.dtype.is_floating_point:
            labels = (pred > 0.5).long(); C_pred = 2
        else:
            labels = pred.long();         C_pred = int(labels.max().item()) + 1 if pred.numel() else 1
        if target.dtype.is_floating_point:
            tgt_labels = (target > 0.5).long(); C_tgt = 2
        else:
            tgt_labels = target.long();         C_tgt = int(tgt_labels.max().item()) + 1 if target.numel() else 1
        C_final = num_classes if num_classes is not None else max(C_pred, C_tgt)
        y_pred = F.one_hot(labels.clamp_min(0), num_classes=C_final).movedim(-1, 1).float()
        y_true = F.one_hot(tgt_labels.clamp_min(0),  num_classes=C_final).movedim(-1, 1).float()
    else:
        raise ValueError(f"Unsupported shapes: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    y_pred = y_pred.to(device)
    y_true = y_true.to(device)

    # ---- MONAI metric: keep background to preserve C channels ----
    hd_metric = HausdorffDistanceMetric(
        include_background=True,          # <-- keep all C channels
        percentile=percentile,
        directed=directed,
        reduction="none",                 # [B,C]
        get_not_nans=False,
    )
    hd = hd_metric(y_pred, y_true, spacing=spacing)     # [B,C] with NaNs for undefined
    hd_per_class = torch.nanmean(hd, dim=0)             # [C]

    # per-class dict
    scores: Dict[str, float] = {
        f"HD{percentile}_class_{c}": float(hd_per_class[c].item()) for c in range(hd_per_class.numel())
    }

    # mean (optionally excluding background), ignore NaNs
    fg = hd_per_class[1:] if (ignore_background and hd_per_class.numel() > 1) else hd_per_class
    scores["mean"] = 0.0 if torch.isnan(fg).all() else float(torch.nanmean(fg).item())
    return scores["mean"]
