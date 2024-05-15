'''
    @desc: This code defines a custom dataset class for change detection tasks using PyTorch's Dataset
      and DataLoader classes. It handles loading and transforming images from a directory structure,
        and it provides utilities for creating data loaders for training, validation, and testing.
'''

import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2 as transforms

from typing import Tuple

IMAGE_PATHS_FILE = "paths.pkl"

# inherit from the torch class DataSet.
class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None, delta_transform=None, is_train = True):
        '''
            Inputs: 
                data_dir: The root directory containing the images.
                transform: Transformations applied to the 'A' and 'B' images.
                delta_transform: Transformations applied to the 'delta' images.
                is_train: A flag indicating whether the dataset is for training or testing.
            Data Members: 
                A_image_paths: Paths to images in the 'A' directory.
                B_image_paths: Paths to images in the 'B' directory.
                delta_image_paths: Paths to images in the 'delta' directory (only if is_train is True).
        '''
        self.transform = transform
        self.delta_transform = delta_transform
        self.is_train = is_train
        
        self.A_image_paths = self._list_images(data_dir, 'A')
        self.B_image_paths = self._list_images(data_dir, 'B')
        if self.is_train:
            self.delta_image_paths = self._list_images(data_dir, 'delta')
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            This loads abd returns a data sample given an index.
            If is_train is True, it returns the transformed 'A', 'B', and 'delta' images.
            If is_train is False, it returns only the transformed 'A' and 'B' images.   
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        A = Image.open(self.A_image_paths[idx])
        B = Image.open(self.B_image_paths[idx])
                
        A = self.transform(A)
        B = self.transform(B)
        
        if self.is_train:
            delta = Image.open(self.delta_image_paths[idx])
            delta = self.delta_transform(delta)
            return A, B, delta
        else:
            return A, B
    
    
    def __len__(self) -> int:
        '''
           Returns the total number of samples in the dataset.
        '''
        return len(self.A_image_paths)
    
    def _list_images(self, data_dir, image_dir):
        '''
            Helper function to list image paths
        '''
        if self.is_train:
            with open(IMAGE_PATHS_FILE, "rb") as file:
                image_list = pickle.load(file)
        else:
           image_list = sorted(os.listdir(os.path.join(data_dir, image_dir)))
            
        image_paths = [os.path.join(data_dir, image_dir, img) for img in image_list]
        return image_paths
    
# Resizze images to 256 x 256 pixels, and convert them to tensors.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Resizze images to 256 x 256 pixels, and convert them to tensors.
# but we also convert them to gray scale. 
delta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.ToDtype(torch.uint8),
    # transforms.Lambda(lambda x: x.squeeze()),
])

transform_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])


def create_data_loaders(data_dir, val_ratio, batch_size, grayscale=False):
    '''
        Creates and returns data loaders for training and validation.
        Inputs: 
            val_ratio: The proportion of the dataset to use for validation.
            batch_size: The number of samples per batch.
            grayscale: A flag indicating whether to convert images to grayscale.
    '''
    image_transform = transform if not grayscale else transform_gray
    dataset = ChangeDetectionDataset(data_dir, image_transform, delta_transform)

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42) # Fix the generator to reproduce the split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def create_test_loader(data_dir, batch_size):
    test_set = ChangeDetectionDataset(data_dir, transform, None, is_train=False)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    return test_loader