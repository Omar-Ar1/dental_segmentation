# stage1_segmentation/utils/polygon_to_mask.py

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from PIL import ImageDraw, Image
import json
import cv2

def polygon_to_mask(polygons, image_size):
    """
    Converts a list of polygons (COCO format) into a binary mask.
    """
    mask = Image.new("L", image_size, 0)
    for polygon in polygons:
        if len(polygon) >= 6:  # At least 3 points (x,y)
            xy = list(zip(polygon[0::2], polygon[1::2]))
            ImageDraw.Draw(mask).polygon(xy, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def parse_coco_annotations(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    anns = pd.DataFrame(coco["annotations"]).rename(columns={"id": "ann_id"})
    imgs = pd.DataFrame(coco["images"]).rename(columns={"id": "image_id"})

    merged = anns.merge(imgs, on="image_id")
    return merged  # Contains file_name, segmentation, bbox, image size, etc.



def overlay(image, mask, color=(0, 255, 0), alpha=0.5, path='Segmentation_Overlay.png'):
    """
    Overlay a binary mask on an image with a specified color.
    
    Args:
        image (np.ndarray): [H,W,3] uint8 image in BGR format
        mask (np.ndarray): [H,W] binary mask (0 or 1)
        color (tuple): BGR color for the mask overlay
        alpha (float): transparency factor
    """
    # Ensure mask is [H,W] and binary
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8) * 255

    # Resize mask if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


    # Ensure mask is binary and uint8
    mask = (mask > 0).astype(np.uint8) * 255

    # Resize mask if needed
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * (color[i] / 255.0)

    # Blend image and mask
    overlay_img = cv2.addWeighted(image, 1.0, colored_mask.astype(np.uint8), alpha, 0)

    # Convert BGR to RGB for matplotlib
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_rgb)
    plt.title('Colored Segmentation Overlay')
    plt.axis('off')
    plt.savefig(path)
    plt.show()
