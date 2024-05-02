import os
import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2 as transforms

from typing import Tuple

IMAGE_PATHS_FILE = "paths.pkl"

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None, delta_transform=None):
        self.transform = transform
        self.delta_transform = delta_transform
        
        self.A_image_paths = self._list_images(data_dir, 'A')
        self.B_image_paths = self._list_images(data_dir, 'B')
        self.delta_image_paths = self._list_images(data_dir, 'delta')
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        A = Image.open(self.A_image_paths[idx])
        B = Image.open(self.B_image_paths[idx])
        delta = Image.open(self.delta_image_paths[idx])
        
        print(self.A_image_paths[idx])
        print(self.B_image_paths[idx])
        print(self.delta_image_paths[idx])
                
        A = self.transform(A)
        B = self.transform(B)
        delta = self.delta_transform(delta)
                    
        return A, B, delta
    
    
    def __len__(self) -> int:
        return len(self.A_image_paths)
    
    def _list_images(self, data_dir, image_dir):
        # image_list = sorted(os.listdir(dir))
        with open(IMAGE_PATHS_FILE, "rb") as file:
            image_list = pickle.load(file)
            
        image_paths = [os.path.join(data_dir, image_dir, img) for img in image_list]
        return image_paths
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

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
    image_transform = transform if not grayscale else transform_gray
    dataset = ChangeDetectionDataset(data_dir, image_transform, delta_transform)

    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
