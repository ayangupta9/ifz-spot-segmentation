import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from lightning.pytorch import LightningDataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

def calculate_class_weights(dataset, num_classes,smoothing_factor=0.1):
    """
    Calculate class weights based on the frequency of each class in the dataset.

    Args:
        dataset (Dataset): Dataset object with images and masks.
        num_classes (int): Number of unique classes.

    Returns:
        torch.Tensor: Class weights.
    """
    class_counts = np.zeros(num_classes)
    
    for _, mask in dataset:
        mask_np = mask.numpy()
        for i in range(num_classes):
            class_counts[i] += np.sum(mask_np == i)

    total_counts = np.sum(class_counts)
    class_weights = total_counts / (num_classes * class_counts)

    softened_weights = smoothing_factor * class_weights + (1 - smoothing_factor)

    return torch.tensor(softened_weights, dtype=torch.float32)


class VegetationDataset_v1(Dataset):
    def __init__(self, image_patches, mask_patches, transform=None):
        """
        Args:
            image_patches (numpy array): Numpy array of shape (1102, 128, 128, 5) representing image data.
            mask_patches (numpy array): Numpy array of shape (1102, 128, 128) representing the corresponding masks.
            transform (callable, optional): Optional transform to be applied on a sample (both image and mask).
        """
        self.image_patches = image_patches
        self.mask_patches = mask_patches
        self.transform = transform

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        image = self.image_patches[idx]
        mask = self.mask_patches[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()



class VegetationDataModule(LightningDataModule):
    def __init__(self, image_patches, mask_patches, batch_size=32, val_split=0.2, num_workers=0):
        super().__init__()
        self.image_patches = image_patches
        self.mask_patches = mask_patches
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        # Define augmentations
        self.train_transform = A.Compose([
            # A.Resize(height=256,width=256, interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_CUBIC, p=1.0),
            A.RandomRotate90(),
            A.Flip(),
            A.RandomBrightnessContrast(),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            # A.Resize(height=256,width=256, interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_CUBIC, p=1.0),
            ToTensorV2()
        ])
        print('Init done')
        
    def setup(self, stage=None):
        dataset_size = len(self.image_patches)
        indices = list(range(dataset_size))
        train_indices, val_indices = train_test_split(indices, test_size=self.val_split, random_state=42)

        self.train_dataset = VegetationDataset_v1(self.image_patches[train_indices], 
                                                  self.mask_patches[train_indices], 
                                                  transform=self.train_transform)
        
        self.val_dataset = VegetationDataset_v1(self.image_patches[val_indices], 
                                                self.mask_patches[val_indices], 
                                                transform=self.val_transform)
        
        num_classes = 4  # Adjust based on your dataset
        self.class_weights = calculate_class_weights(self.train_dataset, num_classes)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
