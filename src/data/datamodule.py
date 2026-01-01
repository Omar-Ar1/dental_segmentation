import os
import lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.components.dental_dataset import DentalSegmentationDataset, _load_coco_grouped
from monai.transforms import Compose, ResizeD, ToTensorD, RandAffineD, RandAdjustContrastD

class DentalDataModule(L.LightningDataModule):
    def __init__(
        self,
        json_path: str,
        img_dir: str,
        input_size: int = 512,
        batch_size: int = 4,
        num_workers: int = 2,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ids_train = []
        self.ids_val = []
        self.ids_test = []

    def setup(self, stage=None):
        # Load all IDs
        images, order = _load_coco_grouped(self.hparams.img_dir, self.hparams.json_path)
        all_ids = list(order)
        
        # Split logic
        train_idx, temp_idx = train_test_split(
            all_ids, 
            test_size=self.hparams.val_frac + self.hparams.test_frac, 
            random_state=self.hparams.seed
        )
        
        # Adjust test size relative to the temp split
        relative_test_frac = self.hparams.test_frac / (self.hparams.val_frac + self.hparams.test_frac)
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=relative_test_frac, 
            random_state=self.hparams.seed
        )
        
        self.ids_train = train_idx
        self.ids_val = val_idx
        self.ids_test = test_idx

        # Define Transforms
        base_transforms = [
             ResizeD(keys=["image", "mask"], spatial_size=(self.hparams.input_size, self.hparams.input_size)),
             ToTensorD(keys=["image", "mask"]),
        ]
        
        # Add augmentations for training
        self.train_transform = Compose([
            RandAffineD(
                keys=['image', 'mask'], 
                rotate_range=(0.15, 0.15), 
                prob=0.5, 
                mode=("bilinear", "nearest")
            ),
            RandAdjustContrastD(keys=["image"], prob=0.5),
            *base_transforms
        ])
        
        self.eval_transform = Compose(base_transforms)

    def train_dataloader(self):
        ds = DentalSegmentationDataset(
            self.hparams.json_path, self.hparams.img_dir, 
            transform=self.train_transform, ids=self.ids_train
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True, 
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = DentalSegmentationDataset(
            self.hparams.json_path, self.hparams.img_dir, 
            transform=self.eval_transform, ids=self.ids_val
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, 
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        ds = DentalSegmentationDataset(
            self.hparams.json_path, self.hparams.img_dir, 
            transform=self.eval_transform, ids=self.ids_test
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, 
                          num_workers=self.hparams.num_workers, pin_memory=True)