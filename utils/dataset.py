from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from utils.transforms import *

__all__= ['get_cifar100_iterator']

# Categories of augmentation for CIFAR100C
CORRUPTED_CATEGORIES = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog",
    "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise",
    "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise",
    "snow", "spatter", "speckle_noise", "zoom_blur"
]

class CIFAR100C(Dataset):
    """
    CIFAR100C Test Dataset: https://arxiv.org/abs/1903.12261
    """
    def __init__(self, category="fog"):
        assert category in CORRUPTED_CATEGORIES + ['all']
        if category == 'all':
            self.data = np.concatenate([np.load(f"./data/CIFAR-100-C/{category}.npy") for category in CORRUPTED_CATEGORIES], axis=0)
            self.targets = np.tile(np.load("./data/CIFAR-100-C/labels.npy"), reps=len(CORRUPTED_CATEGORIES))
        else:
            self.data = np.load(f"./data/CIFAR-100-C/{category}.npy")
            self.targets = np.load("./data/CIFAR-100-C/labels.npy")
        
        self.transform = test_transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)


def baseline_cifar100_dataloader(batch_size=64, num_workers=8):
    train_data = CIFAR100(root="./data", download=True, train=True, transform=train_transform)
    test_data = CIFAR100(root="./data", train=False, transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    return train_dataloader, test_dataloader

def cifar100c_dataloader(batch_size=64, num_workers=8, category='all'):
    dataset = CIFAR100C(category=category)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    cifar100c_dataloader()