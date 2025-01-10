import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.masks = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        self.transform = transform
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Convert mask to tensor without normalization
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

# Separate transforms for images and masks
image_transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = Compose([
    Resize((512, 512)),
    ToTensor()  # No normalization for masks!
])

# Create datasets with correct transforms
train_dataset = SegmentationDataset(
    images_dir="data/train/images",
    masks_dir="data/train/masks",
    transform=image_transform,
    mask_transform=mask_transform  # Pass separate mask transform
)
    
val_dataset = SegmentationDataset(
    images_dir="data/val/images",
    masks_dir="data/val/masks",
    transform=image_transform,
    mask_transform=mask_transform  # Pass separate mask transform
)
    

from torch.utils.data import DataLoader

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
