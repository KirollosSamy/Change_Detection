from src.datasets.dataloader import ChangeDetectionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.GaussianBlur(kernel_size=7, sigma=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0], std=[1]),
])

delta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def create_classical_loader(data_dir, batch_size, grayscale=False):
    dataset = ChangeDetectionDataset(data_dir, transform, delta_transform, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader