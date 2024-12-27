    
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

  

class SICETrainDataset(BaseDataset):
    def __init__(self, root_dir, transform=None, augmentation=None, exposure_type="both"):
        self.root_dir = root_dir
        self.transform = transform
        self.augmentation = augmentation
        self.exposure_type = exposure_type
        self.data = []

        part_path = os.path.join(root_dir, "Dataset_Part2/Dataset_Part2")
        label_path = os.path.join(part_path, "Label")

        for folder in os.listdir(part_path):
            folder_path = os.path.join(part_path, folder)

            if folder.isdigit() and os.path.isdir(folder_path):
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
            half_index = num_images // 2 +1

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
    def __init__(self, root_dir,aug, exposure_type="over", indices=None):
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
        if self.augmentation:
            augmented = self.augmentation(image1=label_image, image=input_image)
            label_image, input_image = augmented['image1'], augmented['image']

        input_image = input_image / 255.0
        label_image = label_image / 255.0

        # CHANGING DIMSPACE
        input_image = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        label_image = torch.tensor(label_image, dtype=torch.float32).permute(2, 0, 1)

        return input_image, label_image

