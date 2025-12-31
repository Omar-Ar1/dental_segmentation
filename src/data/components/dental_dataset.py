# stage1_segmentation/dataset/dental_seg_dataset.py

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms
from monai.transforms import ToTensor
# Your polygon utility: expected to handle a list of polygons and return (H,W) mask
from src.utils.visualization import polygon_to_mask
from monai.data import MetaTensor
# Optional COCO RLE support (only if available)
try:
    import pycocotools.mask as maskUtils  # type: ignore
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False


def _segmentation_to_mask(segmentation: Any, size: Tuple[int, int]) -> np.ndarray:
    """
    Convert COCO-style `segmentation` into a binary mask of shape (H, W).
    Supports:
      - Polygons: list of lists (each sub-list a polygon)
      - RLE dict: {'counts': ..., 'size': [H, W]} if pycocotools is installed

    Args:
        segmentation: COCO 'segmentation' field (polygons or RLE)
        size: (width, height) in pixels

    Returns:
        np.uint8 mask with values {0, 1}
    """
    W, H = size
    if isinstance(segmentation, list):
        # Polygon format
        m = polygon_to_mask(segmentation, (W, H))  # expects (W,H)
        return (m > 0).astype(np.uint8)

    if isinstance(segmentation, dict) and "counts" in segmentation:
        if not _HAS_COCO:
            raise RuntimeError(
                "RLE segmentation found but pycocotools is not installed. "
                "Install pycocotools or convert segmentations to polygons."
            )
        rle = segmentation
        # Ensure RLE has correct size (H, W)
        if "size" not in rle:
            rle = dict(rle)  # shallow copy
            rle["size"] = [H, W]
        m = maskUtils.decode(rle)  # (H, W) array {0,1}
        return (m > 0).astype(np.uint8)

    raise ValueError("Unsupported segmentation format; expected polygons (list) or RLE (dict).")


def _load_coco_grouped(image_dir, json_path: str) -> Tuple[Dict[int, Dict[str, Any]], List[int]]:
    """
    Load COCO JSON and group annotations per image_id.

    Returns:
        images: dict keyed by image_id with:
            {
                'file_name': str,
                'width': int,
                'height': int,
                'anns': List[annotation_dict]
            }
        order: list of image_ids to preserve iteration order
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    
    # Map image_id -> image info
    images: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    files : List[str] = []
    for im in coco["images"]:
        iid = int(im["id"])
        file_path = os.path.join(image_dir, im["file_name"])
        if not os.path.isfile(file_path):
            continue  # skip missing files
        images[iid] = {
            "file_name": im["file_name"],
            "width": int(im.get("width", 0) or 0),
            "height": int(im.get("height", 0) or 0),
            "anns": [],
        }
        order.append(iid)
        files.append(im['file_name'])

    # Attach annotations
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        if iid not in images:
            # Or skip silently if dangling annotation
            print(f'{iid} not in Images')
            continue
        images[iid]["anns"].append(ann)

    # Sort `order` according to the sorted order of `files`
    order = [x for _, x in sorted(zip(files, order), key=lambda pair: pair[0])]

    return images, order


def _pad_image(img, patch_size=14):
    W, H = img.size  # PIL uses (width, height)

    pad_H = (patch_size - H % patch_size) % patch_size
    pad_W = (patch_size - W % patch_size) % patch_size

    # Padding: (left, top, right, bottom)
    pad_left = pad_W // 2
    pad_right = pad_W - pad_left
    pad_top = pad_H // 2
    pad_bottom = pad_H - pad_top

    pad = transforms.Pad((pad_left, pad_top, pad_right, pad_bottom))
    return pad(img)



class DentalSegmentationDataset(Dataset):
    """
    Full-image dental segmentation dataset.

    Default behavior is **binary masks** (background=0, tooth=1) formed by
    union of all tooth polygons per image. This matches CrossEntropyLoss with
    num_classes=2 (logits shape [B,2,H,W], target [B,H,W] as indices).

    If you later want multi-class masks (e.g., per-category_id), update:
      - `mask_mode='category'`
      - Ensure your paired ToTensor keeps integer labels (does NOT binarize).
    """

    def __init__(
        self,
        json_path: str,
        img_dir: str,
        transform: Optional[Any] = None,
        mask_mode: str = "binary",  # 'binary' | 'category' | 'instance' (binary is default/expected)
        category_id_to_index: Optional[Dict[int, int]] = None,
        drop_images_without_annotations: bool = False,
        ids: Optional[List[int]] = None,
    ):
        """
        Args:
            json_path: path to COCO JSON
            img_dir: root folder with images
            transform: paired transform (img, mask) -> (img_t, mask_t)
            mask_mode: 'binary' (default), 'category', or 'instance'
            category_id_to_index: optional mapping from COCO category_id -> class index (for 'category' mode)
            drop_images_without_annotations: if True, skip images with zero anns
        """
        self.img_dir = img_dir
        self.transform = transform
        self.mask_mode = mask_mode
        self.cat_map = category_id_to_index or {}

        self._images, order = _load_coco_grouped(image_dir=img_dir, json_path=json_path)

        if drop_images_without_annotations:
            order = [iid for iid in order if len(self._images[iid]["anns"]) > 0]

        if ids:
            ids_set = set(ids)
            order = [iid for iid in order if iid in ids_set]
            
        self._order = order

        if self.mask_mode not in {"binary", "category", "instance"}:
            raise ValueError("mask_mode must be one of {'binary', 'category', 'instance'}")

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, idx: int):
        iid = self._order[idx]
        entry = self._images[iid]

        img_path = os.path.join(self.img_dir, entry["file_name"])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        anns = entry["anns"]

        # --- build mask ---
        if self.mask_mode == "binary":
            mask_np = np.zeros((H, W), dtype=np.uint8)
            for a in anns:
                seg = a.get("segmentation", None)
                if seg is None:
                    continue
                m = _segmentation_to_mask(seg, (W, H))
                mask_np |= m
            mask_np = mask_np * 255  # 0 or 255

        elif self.mask_mode == "category":
            mask_np = np.zeros((H, W), dtype=np.int32)
            for a in anns:
                seg = a.get("segmentation", None)
                if seg is None:
                    continue
                cat_id = int(a.get("category_id", 0))
                cls = self.cat_map.get(cat_id, cat_id)
                if cls == 0:
                    cls = 1
                m = _segmentation_to_mask(seg, (W, H)).astype(bool)
                mask_np[m] = cls

        else:  # instance mode
            mask_np = np.zeros((H, W), dtype=np.int32)
            inst_id = 0
            for a in anns:
                seg = a.get("segmentation", None)
                if seg is None:
                    continue
                inst_id += 1
                m = _segmentation_to_mask(seg, (W, H)).astype(bool)
                mask_np[m] = inst_id

        # --- Convert image/mask to numpy CHW before MONAI ---
        img_np = np.asarray(img, dtype=np.float32)      # H W C
        img_np = np.transpose(img_np, (2, 0, 1))        # C H W

        mask_np = mask_np.astype(np.int64)[None, ...]   # 1 H W

        sample = {"image": img_np, "mask": mask_np}

        # --- Apply MONAI transforms ---
        if self.transform is not None:
            sample = self.transform(sample)

        # --- Convert MetaTensor → torch.Tensor ---
        img_t = sample["image"]
        mask_t = sample["mask"]

        if isinstance(img_t, MetaTensor):
            img_t = img_t.as_tensor()
        if isinstance(mask_t, MetaTensor):
            mask_t = mask_t.as_tensor()

        mask_t = (mask_t.squeeze(0) > 0).long()
        return img_t, mask_t


    def __repr__(self) -> str:
        return (
            f"DentalSegmentationDataset(n={len(self)}, img_dir='{self.img_dir}', "
            f"mask_mode='{self.mask_mode}')"
        )

