"""Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027

modified code from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["resnet8", "resnet20", "resnet32", "resnet44", "resnet56", 
            "wideresnet16_1", "wideresnet16_2", "wideresnet16_4",
            "wideresnet28_4", "wideresnet40_1", "wideresnet40_2"]

class BasicBlock(nn.Module):
    """
    Pre-activation BasicBlock
    Uses shortcut connection option B.
    BatchNorm is not applied on shortcuts following
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    def __init__(self, in_filters, filters, stride=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_filters)
        self.conv1 = nn.Conv2d(in_filters, filters, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)

        # projection shortcut when changing dims (option B)
        if stride != 1 or in_filters != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_filters, filters, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    """
    ResNet variant for CIFAR (3 blocks) as described in https://arxiv.org/pdf/1512.03385.pdf

    Also supports wide resnet models with scaling parameter k. 
    """
    def __init__(self, block, filters=[16, 32, 64], n=3, k=1, num_classes=100):
        super(ResNet, self).__init__()
        self.in_filters = 16
        self.k = k
        self.layer0 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, k*filters[0], n, stride=1)
        self.layer2 = self._make_layer(block, k*filters[1], n, stride=2)
        self.layer3 = self._make_layer(block, k*filters[2], n, stride=2)
        self.finalact = nn.Sequential(nn.BatchNorm2d(k*filters[2]), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear = nn.Linear(k*filters[2], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, filters, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_filters, filters, stride))
            self.in_filters = filters
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        x = self.finalact(layer3)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return [layer1, layer2, layer3], x


def resnet8(num_classes=100, model_path=None):
    model = ResNet(block=BasicBlock, n=1, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def resnet20(num_classes=100, model_path=None):
    model = ResNet(block=BasicBlock, n=3, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def resnet32(num_classes=100, model_path=None):
    model = ResNet(block=BasicBlock, n=5, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def resnet44(num_classes=100, model_path=None):
    model = ResNet(block=BasicBlock, n=7, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def resnet56(num_classes=100, model_path=None):
    model = ResNet(block=BasicBlock, n=9, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet16_1(num_classes=100, model_path=None):
    n = (16 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=1, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet16_2(num_classes=100, model_path=None):
    n = (16 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=2, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet16_4(num_classes=100, model_path=None):
    n = (16 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=4, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet28_4(num_classes=100, model_path=None):
    n = (28 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=4, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet40_1(num_classes=100, model_path=None):
    n = (40 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=1, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model

def wideresnet40_2(num_classes=100, model_path=None):
    n = (40 - 4) // 6
    model = ResNet(block=BasicBlock, n=n, k=2, num_classes=num_classes)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model