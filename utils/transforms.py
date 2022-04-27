import numpy as np
from torchvision import transforms
import cv2

# Statistics from CIFAR100 training set
CIFAR100_mean = np.array([129.30, 124.07, 112.43]) / 255.0
CIFAR100_std = np.array([68.17, 65.39, 70.42]) / 255.0

# Baseline augmentation policy for CIFAR100
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

def saliency_bbox(img, lam):
    """
    Extract salient bounding box
    https://github.com/afm-shahab-uddin/SaliencyMix/blob/main/SaliencyMix_CIFAR/saliencymix.py
    """
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and
    # compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliencyMap, axis=None), saliencyMap.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2