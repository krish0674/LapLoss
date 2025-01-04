    
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

import albumentations as albu
import cv2

def get_training_augmentation():
    train_transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
        albu.VerticalFlip(p=0.3),
        albu.HorizontalFlip(p=0.3),
        albu.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.3),
        albu.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        albu.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        albu.GaussianBlur(blur_limit=(3, 5), p=0.3),
        # albu.GridDistortion(num_steps=3, distort_limit=0.01, p=0.3),
    ]
    return albu.Compose(train_transform, additional_targets={'image1': 'image'}, is_check_shapes=False)

def get_testing_augmentation():
    transform = [
        albu.Resize(608, 896, interpolation=cv2.INTER_LINEAR, always_apply=True),
    ]
    return albu.Compose(transform, additional_targets={'image1': 'image'}, is_check_shapes=False)

  

import os
import cv2
import random
import torch
from torch.utils.data import Dataset

class SICETrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, augmentation=None, exposure_type="both", split_type="train", split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.exposure_type = exposure_type
        self.split_type = split_type
        self.data = []

        random.seed(seed)

        part_path = os.path.join(root_dir, "Dataset_Part2/Dataset_Part2")
        label_path = os.path.join(part_path, "Label")

        folders = [
            folder for folder in os.listdir(part_path) 
            if folder.isdigit() and os.path.isdir(os.path.join(part_path, folder))
        ]

        # Set seed for reproducibility and shuffle
        random.seed(seed)
        random.shuffle(folders)

        # Split folders into train and validation sets
        split_index = int(len(folders) * split_ratio)
        train_folders = folders[:split_index]
        val_folders = folders[split_index:]

        selected_folders = train_folders if split_type == "train" else val_folders

        for folder in selected_folders:
            folder_path = os.path.join(part_path, folder)

            # Check for either .png or .jpg label file
            label_file = None
            for ext in [".PNG", ".JPG", ".JPEG"]:
                potential_label = os.path.join(label_path, f"{folder}{ext}")
                if os.path.exists(potential_label):
                    label_file = potential_label
                    break

            if not label_file:
                continue  # Skip if no valid label file found

            # Add valid image-label pairs
            image_files = [
                os.path.join(folder_path, img_file)
                for img_file in sorted(os.listdir(folder_path))
                if img_file.endswith((".PNG", ".JPG", ".JPEG"))
            ]
            self.data.append((image_files, label_file))

        # Filter images based on exposure type
        filtered_data = []
        for image_files, label_file in self.data:
            num_images = len(image_files)
            half_index = num_images // 2 + 1

            if self.exposure_type == "under":
                filtered_data.extend([(img, label_file) for img in image_files[:half_index]])
            elif self.exposure_type == "over":
                filtered_data.extend([(img, label_file) for img in image_files[half_index:]])
            elif self.exposure_type == "both":
                filtered_data.extend([(img, label_file) for img in image_files])

        self.data = filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]

        label_image = cv2.imread(label_path)
        input_image = cv2.imread(img_path)

        # CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # RESIZE + FLIP AUGMENTATION
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']

        # NORMALIZATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0

        # CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)

        return input_image, label_image

from torch.utils.data import DataLoader, Dataset
import cv2
import os
import torch
import matplotlib.pyplot as plt

class SICETestDataset(Dataset):
    def __init__(self, root_dir, exposure_type="over", indices=None):
        self.root_dir = root_dir
        self.exposure_type = exposure_type
        self.indices = indices if indices else []
        self.data = []
        self.augumentation=get_testing_augmentation()

        part_path = os.path.join(root_dir, "Dataset_Part1/Dataset_Part1")
        label_path = os.path.join(part_path, "Label")

        for folder in os.listdir(part_path):
            folder_path = os.path.join(part_path, folder)

            if folder.isdigit() and os.path.isdir(folder_path):
                folder_num = int(folder)
                if folder_num not in self.indices:
                    continue

                # Check for label file
                label_file = None
                for ext in [".PNG", ".JPG", ".JPEG"]:
                    potential_label = os.path.join(label_path, f"{folder}{ext}")
                    if os.path.exists(potential_label):
                        label_file = potential_label
                        break

                if not label_file:
                    continue  # Skip if no valid label file found

                # Get all image files in the folder
                image_files = [
                    os.path.join(folder_path, img_file)
                    for img_file in sorted(os.listdir(folder_path))
                    if img_file.endswith((".PNG", ".JPG", ".JPEG"))
                ]

                # Determine the index for the specified exposure type
                num_images = len(image_files)
                if self.exposure_type == "under":
                    if num_images == 7:
                        idx = 2
                    elif num_images == 9:
                        idx = 3
                    else:
                        continue
                elif self.exposure_type == "over":
                    if num_images == 7:
                        idx = 4
                    else:
                        continue
                else:
                    raise ValueError("Invalid exposure type. Choose 'under' or 'over'.")

                if 0 <= idx < num_images:
                    self.data.append((image_files[idx], label_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_path = self.data[idx]

        label_image = cv2.imread(label_path)
        input_image = cv2.imread(img_path)

        # CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # NORMALIZATION
        if self.augumentation:
            augmented = self.augumentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']

        # if random.random() < 0.5:
        #     input_image = cv2.flip(input_image, 1)  # Horizontal flip
        #     label_image = cv2.flip(label_image, 1)
        
        # if random.random() < 0.5:
        #     angle = random.uniform(-15, 15)  # Random rotation
        #     h, w = input_image.shape[:2]
        #     center = (w // 2, h // 2)
        #     rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        #     input_image = cv2.warpAffine(input_image, rot_mat, (w, h))
        #     label_image = cv2.warpAffine(label_image, rot_mat, (w, h))
        
        # if random.random() < 0.5:
        #     scale_factor = random.uniform(0.8, 1.2)  # Random scaling
        #     h, w = input_image.shape[:2]
        #     input_image = cv2.resize(input_image, (int(w * scale_factor), int(h * scale_factor)))
        #     label_image = cv2.resize(label_image, (int(w * scale_factor), int(h * scale_factor)))
        
        # if random.random() < 0.3:
        #     input_image = cv2.GaussianBlur(input_image, (5, 5), 0) #Gaussian Blur

        input_image = input_image / 255.0
        label_image = label_image / 255.0

        # CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)

        return input_image, label_image

    
class SICEGradTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Grad')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augumentation=get_testing_augmentation()
        # self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augumentation:
            augmented = self.augumentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
    
class SICEMixTest(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'SICE_Mix')
        self.label_dir = os.path.join(root_dir, 'SICE_Reshape')
        self.file_names = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]
        self.transform = transform
        self.augumentation=get_testing_augmentation()
        # self.file_names = self.file_names[530:]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_name = os.path.join(self.input_dir, self.file_names[idx])
        base_name = os.path.splitext(self.file_names[idx])[0]
        
        label_name_jpg_upper = os.path.join(self.label_dir, base_name + '.JPG')
        label_name_jpg_lower = os.path.join(self.label_dir, base_name + '.jpg')
        label_name_png = os.path.join(self.label_dir, base_name + '.PNG')
        
        if os.path.exists(label_name_jpg_upper):
            label_name = label_name_jpg_upper
        elif os.path.exists(label_name_jpg_lower):
            label_name = label_name_jpg_lower
        elif os.path.exists(label_name_png):
            label_name = label_name_png
        else:
            raise FileNotFoundError(f"No corresponding label image found for {input_name}")
        
        #READING IMAGE
        label_image = cv2.imread(label_name)
        input_image = cv2.imread(input_name)
        
        #CHANGING COLORSPACE
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        #ROTATING PORTRAIT IMAGES
        if label_image.shape[0] > label_image.shape[1]:
            label_image = cv2.rotate(label_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            input_image = cv2.rotate(input_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #RESIZE + FLIP AUGMENTATION
        if self.augumentation:
            augmented = self.augumentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']
        
        #NORMALISATION
        input_image = input_image / 255.0
        label_image = label_image / 255.0
        
        #STANDARDISATION
        # if self.transform:
        #     input_image = self.transform(image=input_image)['image']
        
        #CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)
        
        return input_image, label_image
