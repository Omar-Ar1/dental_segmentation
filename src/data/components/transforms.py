# stage1_segmentation/dataset/transforms.py

from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask


class ToTensor:
    def __call__(self, data):
        img, mask = data["image"], data["mask"]
        return {"image": TF.to_tensor(img), "mask": TF.pil_to_tensor(mask).squeeze(0)}


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, mask = data["image"], data["mask"]
        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return {"image": img, "mask": mask}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
