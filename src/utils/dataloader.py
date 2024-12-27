    
from torch.utils.data import Dataset as BaseDataset
import cv2
import pickle
import os
import numpy as np
import albumentations as albu
import torch
import pandas as pd
from tqdm import tqdm as tqdm
from torch.utils.data import DataLoader, Subset
import random
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
import cv2
import pickle
import os
import numpy as np

import torch
import pandas as pd

def get_training_augmentation():
    train_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.VerticalFlip(p=0.5),
    ]
    return albu.Compose(train_transform, additional_targets={'image1':'image'}, is_check_shapes=False)

def get_testing_augmentation():
    transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ]
    return albu.Compose(transform, additional_targets={'image1': 'image'}, is_check_shapes=False)

  

class RainToNoRainTrain(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.rain_dir = os.path.join(root_dir, 'rain')
        self.norain_dir = os.path.join(root_dir, 'norain')
        self.file_names = [f for f in os.listdir(self.rain_dir) if f.lower().endswith('.png')]
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        rain_name = os.path.join(self.rain_dir, self.file_names[idx])
        norain_name = os.path.join(self.norain_dir, self.file_names[idx])
        
        if not os.path.exists(norain_name):
            raise FileNotFoundError(f"No corresponding norain image found for {rain_name}")
        rain_image = cv2.imread(rain_name)
        norain_image = cv2.imread(norain_name)
        
        if rain_image is None or norain_image is None:
            raise ValueError(f"Error reading images: {rain_name}, {norain_name}")
        
        # Changing colorspace
        rain_image = cv2.cvtColor(rain_image, cv2.COLOR_BGR2RGB)
        norain_image = cv2.cvtColor(norain_image, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.augmentation:
            augmented = self.augmentation(image1=rain_image, image2=norain_image)
            rain_image, norain_image = augmented['image1'], augmented['image2']
        
        # Transformations
        if self.transform:
            rain_image = self.transform(image=rain_image)['image']
            norain_image = self.transform(image=norain_image)['image']
        else:
            # Normalization
            rain_image = rain_image / 255.0
            norain_image = norain_image / 255.0
            
            # Changing dimspace
            rain_image = torch.tensor(rain_image, dtype=torch.float32).permute(2, 0, 1)
            norain_image = torch.tensor(norain_image, dtype=torch.float32).permute(2, 0, 1)
        
        return rain_image, norain_image

class RainToNoRainTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.rain_dir = os.path.join(root_dir, 'rain')
        self.norain_dir = os.path.join(root_dir, 'norain')
        self.file_names = [f for f in os.listdir(self.rain_dir) if f.lower().endswith('.png')]
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        rain_name = os.path.join(self.rain_dir, self.file_names[idx])
        norain_name = os.path.join(self.norain_dir, self.file_names[idx])
        
        if not os.path.exists(norain_name):
            raise FileNotFoundError(f"No corresponding norain image found for {rain_name}")
        rain_image = cv2.imread(rain_name)
        norain_image = cv2.imread(norain_name)
        
        if rain_image is None or norain_image is None:
            raise ValueError(f"Error reading images: {rain_name}, {norain_name}")
        
        # Changing colorspace
        rain_image = cv2.cvtColor(rain_image, cv2.COLOR_BGR2RGB)
        norain_image = cv2.cvtColor(norain_image, cv2.COLOR_BGR2RGB)

        # Augmentation
        if self.augmentation:
            augmented = self.augmentation(image1=rain_image, image2=norain_image)
            rain_image, norain_image = augmented['image1'], augmented['image2']
        
        # Transformations
        if self.transform:
            rain_image = self.transform(image=rain_image)['image']
            norain_image = self.transform(image=norain_image)['image']
        else:
            # Normalization
            rain_image = rain_image / 255.0
            norain_image = norain_image / 255.0
            
            # Changing dimspace
            rain_image = torch.tensor(rain_image, dtype=torch.float32).permute(2, 0, 1)
            norain_image = torch.tensor(norain_image, dtype=torch.float32).permute(2, 0, 1)
        
        return rain_image, norain_image
