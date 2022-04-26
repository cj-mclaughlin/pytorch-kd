import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from utils.dataset import cifar100c_dataloader, CORRUPTED_CATEGORIES
from models.resnet import *
import numpy as np

def evaluate_category(model, category='all'):
    """
    Evaluate model category of CIFAR-100C 
    """
    total = 0
    correct = 0
    model.eval()
    
    dataloader = cifar100c_dataloader(category=category)

    for i, (input, target) in enumerate(dataloader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        _, outputs = model(input)
        prediction = outputs.max(1)[1]
        total += target.size(0)
        correct += ( target == prediction ).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def evaluate_cifar100c(model):
    """
    Full evaluation onf cifar100c
    """
    results = {}
    for category in CORRUPTED_CATEGORIES:
        result = evaluate_category(model, category)
        results[category] = result
        print(f"Category {category}: {result}")
    mean = np.mean([results[c] for c in CORRUPTED_CATEGORIES])
    results['all'] = mean
    return results

if __name__ == "__main__":
    model = wideresnet40_2(model_path="models/pretrained/wrn40_2.pth").cuda()
    results = evaluate_cifar100c(model)
    print(results)