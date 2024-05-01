import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2 as transforms

from typing import Tuple

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None, delta_transform=None):
        self.transform = transform
        self.delta_transform = delta_transform
        
        A_dir = os.path.join(data_dir, 'A')
        B_dir = os.path.join(data_dir, 'B')
        delta_dir = os.path.join(data_dir, 'delta')
        
        self.A_image_paths = self._list_images(A_dir)
        self.B_image_paths = self._list_images(B_dir)
        self.delta_image_paths = self._list_images(delta_dir)
        
        # self.A_images = self._load_images(A_dir)
        # self.B_images = self._load_images(B_dir)
        # self.delta_images = self._load_images(delta_dir)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        A = Image.open(self.A_image_paths[idx])
        B = Image.open(self.B_image_paths[idx])
        delta = Image.open(self.delta_image_paths[idx])
                
        A = self.transform(A)
        B = self.transform(B)
        delta = self.delta_transform(delta)
                    
        return A, B, delta
    
    
    def __len__(self) -> int:
        return len(self.A_image_paths)
    
    def _list_images(self, dir):
        image_list = os.listdir(dir)
        image_paths = [os.path.join(dir, img) for img in image_list]
        return image_paths
    
    # def _load_images(self, dir):
    #     image_paths = self._list_images(dir)
    #     images = [Image.open(path) for path in image_paths]
    #     return images
    

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

delta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.ToDtype(torch.uint8),
    transforms.Lambda(lambda x: x.squeeze()),
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
