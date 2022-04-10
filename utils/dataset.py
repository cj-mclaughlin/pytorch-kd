from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

__all__= ['get_cifar100_iterator']

# Statistics from CIFAR100 training set
CIFAR100_mean = np.array([129.30, 124.07, 112.43]) / 255.0
CIFAR100_std = np.array([68.17, 65.39, 70.42]) / 255.0

train_transform = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_mean, std=CIFAR100_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR100_mean, std=CIFAR100_std)
])

def get_cifar100_iterator(batch_size=64, num_workers=8):
    train_data = CIFAR100(root="./data", download=True, train=True, transform=train_transform)
    test_data = CIFAR100(root="./data", train=False, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    return train_loader, test_loader