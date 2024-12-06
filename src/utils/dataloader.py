from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import pickle
import os
import numpy as np
import re
import torch
import pandas as pd


def load_dataset():
    dataset_filename = "ULB17-VT.pkl"  # Update this if the filename is different
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(current_dir,"utils", dataset_filename)
    fileobj = open(dataset_path,'rb')
    myfiles = pickle.load(fileobj)
    return myfiles

def create_dataset(dataset_folder, subfolder_names, data):
    # Create the main dataset folder if it doesn't exist
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Create the 'Test', 'Train', and 'Validation' folders
    test_folder = os.path.join(dataset_folder, 'Test')
    train_folder = os.path.join(dataset_folder, 'Train')
    validation_folder = os.path.join(dataset_folder, 'Validation')

    # Create the 'Test' folder if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Create the 'Train' folder if it doesn't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    # Create the 'Validation' folder if it doesn't exist
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    # Iterate over the subfolder names
    for i, folder_name in enumerate(subfolder_names):
        # Iterate over the 'Train', 'Validation', and 'Test' folders
        for j, split_folder in enumerate(['Train', 'Validation', 'Test']):
            # Specify the path of the current folder
            current_folder = os.path.join(dataset_folder, split_folder, folder_name + '_' + split_folder)

            # Create the current folder if it doesn't exist
            if not os.path.exists(current_folder):
                os.makedirs(current_folder)
            # Save the ndarray data in the current folder
            for k in range(len(data[j][i])):
                file_path = os.path.join(current_folder,f'data_{folder_name}_{split_folder}_{k}.npy')
                np.save(file_path,data[j][i][k])
            


       
class Dataset():  
    
    def __init__(
            self, 
            rgb_dir: str, 
            thermal_low_res_dir:str,
            thermal_high_res_tar_dir: str, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.rgb_list = self.load_img(rgb_dir)
        self.thermal_low_res_list = self.load_img(thermal_low_res_dir)
        self.thermal_high_res_tar_list = self.load_img(thermal_high_res_tar_dir)
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def standardize(self,image,mean,std):
        image = image/255
        # image_normalised = image - mean
        # image_standardized = image_normalised / std
        image_standardized = image
        return image_standardized
    
    def __getitem__(self, i):
        
        # read data
        rgb_image = np.load(self.rgb_list[i])
        target_image = np.load(self.thermal_high_res_tar_list[i])
        thermal_low_res_image = np.load(self.thermal_low_res_list[i])
        gray_image = cv2.cvtColor(np.transpose(rgb_image,(1,2,0)), cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.reshape(1,240,320)
            
        if self.augmentation:
            
            augmented = self.augmentation(image=gray_image, target=target_image, thermal_low_res_image=thermal_low_res_image)
            gray_image,target_image,thermal_low_res_image = augmented['image'],augmented['target'],augmented['thermal_low_res_image']
        
        if self.preprocessing:
            
            gray_image = self.standardize(gray_image,0.38206747,0.22966413)
            target_image = target_image/255.0
            thermal_low_res_image = self.standardize(thermal_low_res_image,0.08869906,0.25463683)
            
        #target_image = normalize_data(target_image)
            
        return gray_image.astype(np.float32),thermal_low_res_image.astype(np.float32),target_image.astype(np.float32)
    
    def load_img(self, directory):
        img_list = []

        directory = directory.lstrip(".")

        files = os.listdir(directory)
        for i in range(len(files)):
            file_path = os.path.join(directory,files[i])
            img_list.append(file_path)
            
        def extract_numerical_value(filename):
            match = re.search(r'_(\d+)\.npy$', filename)
            if match:
                numerical_value = int(match.group(1))
                return numerical_value
            else:
                return None
        img_list = sorted(img_list, key=extract_numerical_value)
        return img_list
    
    def __len__(self):
        return len(self.rgb_list)
    
    
class Dataset_flir():  
    
    def __init__(
            self,
            rgb_dir: str,
            thermal_high_res_tar_dir: str,
            rgb_df = pd.DataFrame,
            thermal_high_res_df = pd.DataFrame,
            augmentation=None, 
            preprocessing=None,
            resize = None,
    ):
        self.rgb_list = self.load_img(rgb_dir,rgb_df)
        self.thermal_high_res_tar_list = self.load_img(thermal_high_res_tar_dir,thermal_high_res_df)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize = resize
        
    def standardize(self,image,mean,std):
        image = image/255
        # image_normalised = image - mean
        # image_standardized = image_normalised / std
        image_standardized = image
        
        return image_standardized
    
    def __getitem__(self, i):
        
        # read data
        
        
        gray_image = cv2.imread(self.rgb_list[i],cv2.IMREAD_GRAYSCALE)
        target_image = cv2.imread(self.thermal_high_res_tar_list[i], 0)

            
        if self.augmentation:
            
            augmented = self.augmentation(image=gray_image, target=target_image)
            gray_image,target_image = augmented['image'],augmented['target']
            
            if self.resize:
                transform = self.resize(image = target_image)
                thermal_low_res_image = transform['image']
                thermal_low_res_image = np.array(thermal_low_res_image)
        
                
        target_image = np.array(target_image)
        
        if self.preprocessing:
            
            gray_image = self.standardize(gray_image,0.39510894298270915,0.24916220656101468)
            target_image = target_image/255.0
            thermal_low_res_image = self.standardize(thermal_low_res_image,0.5091382143477134,0.2146979818912373)

        gray_image = torch.from_numpy(np.expand_dims(gray_image, axis=0)).float()
        target_image = torch.from_numpy(np.expand_dims(target_image, axis=0)).float()
        thermal_low_res_image = torch.from_numpy(np.expand_dims(thermal_low_res_image, axis=0)).float()

        return gray_image, thermal_low_res_image, target_image
    
    def load_img(self,directory,df):
        img_list = []
        directory = directory.lstrip('.')
            
        for i in range(len(df)):
            file_path = os.path.join(directory,df[i])
            img_list.append(file_path)
        return img_list
    
    def __len__(self):
        return len(self.rgb_list)
    
    
class Dataset_cats():  
    
    def __init__(
            self, 
            rgb_list: list, 
            thermal_high_res_list: list, 
            augmentation=None, 
            preprocessing=None,
            resize = None,
    ):
        self.rgb_list = rgb_list
        self.thermal_high_res_list = thermal_high_res_list
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize = resize
        
    def standardize(self,image):
        image = image/255
        return image
    
    def __getitem__(self, i):
        
        # read data
        rgb_image = cv2.imread(self.rgb_list[i])
        target_image = cv2.imread(self.thermal_high_res_list[i],0)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        if self.augmentation:
            
            augmented = self.augmentation(image=gray_image, target=target_image)
            gray_image,target_image = augmented['image'],augmented['target']
            
            if self.resize:
                transform = self.resize(image = target_image)
                thermal_low_res_image = transform['image']
                thermal_low_res_image = np.array(thermal_low_res_image)
        
                
        target_image = np.array(target_image)
        
        if self.preprocessing:
            
            gray_image = self.standardize(gray_image)
            target_image = target_image/255.0
            thermal_low_res_image = self.standardize(thermal_low_res_image)
            
        gray_image = np.expand_dims(gray_image, axis=0)
        target_image = np.expand_dims(target_image, axis=0)
        thermal_low_res_image = np.expand_dims(thermal_low_res_image, axis=0)
        
        return gray_image.astype(np.float32),thermal_low_res_image.astype(np.float32),target_image.astype(np.float32)
    
    def __len__(self):
        return len(self.rgb_list)
            

def create_list2(foldername, fulldir=True, suffix=".png", multifolder=True, ):
    file_list_tmp = os.listdir(foldername)
    file_list_tmp.sort()
    thermal_files_list = []
    total_color = []
    total_thermal = []
    if fulldir:
        if multifolder:
            for folder in file_list_tmp:
                temp_list = os.listdir(os.path.join(foldername, folder))
                temp_list.sort()
                for item in temp_list:
                    temp_list2 = os.listdir(os.path.join(foldername,folder,item))
                    temp_list2.sort()
                    for item2 in temp_list2:
                        temp_list3 = os.listdir(os.path.join(foldername,folder,item,item2))
                        temp_list3.sort()
                        for item3 in temp_list3:
                            if item3.endswith('.XYZ') or item3.endswith('.ply'):
                                pass
                            else:
                                if item3 == 'rectified':
                                    temp_list4 = os.listdir(os.path.join(foldername,folder,item,item2,item3))
                                    temp_list4.sort()
                                    for item4 in temp_list4:
                                        if item4 == 'cross':
                                            temp_list5 = os.listdir(os.path.join(foldername,folder,item,item2,item3,item4))
                                            thermal_files = sorted([name for name in temp_list5 if "thermal" in name])
                                            sorted_color = sorted(sorted([name for name in temp_list5 if "color" in name]), key=lambda x: not x.startswith("right"))
                                            thermal_files_with_paths = [os.path.join(foldername, folder, item, item2, item3,item4,file_name) for file_name in thermal_files]
                                            sorted_color_with_paths = [os.path.join(foldername, folder, item, item2, item3,item4,file_name) for file_name in sorted_color]
                                            total_thermal.extend(thermal_files_with_paths)
                                            total_color.extend(sorted_color_with_paths)

    
    return total_color,total_thermal  

class Dataset_UAV():  
    
    def __init__(
            self, 
            rgb_list: list, 
            thermal_low_res_list: list,
            thermal_high_res_tar_list: list, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.rgb_list = rgb_list
        self.thermal_low_res_list = thermal_low_res_list
        self.thermal_high_res_tar_list = thermal_high_res_tar_list
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def standardize(self,image):
        image = image/255
        image_standardized = image
        return image_standardized
    
    
    def __getitem__(self, i):
        
        # read data
        rgb_image = cv2.imread(self.rgb_list[i])
        target_image = cv2.imread(self.thermal_high_res_tar_list[i])
        thermal_low_res_image = cv2.imread(self.thermal_low_res_list[i])
            
        if self.augmentation:
            
            augmented = self.augmentation(image=rgb_image, target=target_image, thermal_low_res_image=thermal_low_res_image)
            rgb_image,target_image,thermal_low_res_image = augmented['image'],augmented['target'],augmented['thermal_low_res_image']
        
        if self.preprocessing:
            
            rgb_image = self.standardize(rgb_image)
            target_image = target_image/255.0
            thermal_low_res_image = self.standardize(thermal_low_res_image)
            
        #target_image = normalize_data(target_image)
        rgb_image = np.transpose(rgb_image,(2,0,1))
        target_image = np.transpose(target_image,(2,0,1))
        thermal_low_res_image = np.transpose(thermal_low_res_image,(2,0,1))
            
        return rgb_image.astype(np.float32),thermal_low_res_image.astype(np.float32),target_image.astype(np.float32)
    
    def __len__(self):
        return len(self.rgb_list)


def get_sorted_file_paths(directory):
    files = os.listdir(directory)
    img_list = [os.path.join(directory, file) for file in files]
    return sorted(img_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

def sort_files_by_name(thermal_high_res_dir, thermal_low_res_dir, rgb_dir):
    thermal_high_res_sorted_list = get_sorted_file_paths(thermal_high_res_dir)
    thermal_low_res_sorted_list = get_sorted_file_paths(thermal_low_res_dir)
    rgb_sorted_list = get_sorted_file_paths(rgb_dir)
    
    return thermal_high_res_sorted_list, thermal_low_res_sorted_list, rgb_sorted_list