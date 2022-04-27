"""
SaliencyMix Training https://arxiv.org/abs/2006.01791
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from utils.dataset import baseline_cifar100_dataloader
from utils.optimizer import get_default_optim, decay_lr
from utils.transforms import saliency_bbox
from models.resnet import *
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True

# default parameters from saliencymix repo
MIX_PROB = 0.5
BETA = 1

def main(model, num_epochs=240, save_path=""):
    optimizer = get_default_optim(model)
    train_dataloader, test_dataloader = baseline_cifar100_dataloader()
    for epoch in range(num_epochs):
        train(model, train_dataloader, optimizer, epoch+1)
        if (epoch+1) % 30 == 0:
            evaluate(model, test_dataloader)
    print("Final evaluation:")
    final_acc = evaluate(model, test_dataloader)
    torch.save(model.state_dict(), save_path)
    return final_acc

def train(model, dataloader, optimizer, epoch=0):
    model.train()
    criterion = CrossEntropyLoss().cuda()
    
    i = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for input, target in tepoch:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()
            r = np.random.rand(1)
            if r > MIX_PROB:
                # generate mixed sample
                lam = np.random.beta(BETA, BETA)
                rand_index = torch.randperm(input.size()[0]).cuda()
                labels_a = target
                labels_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = saliency_bbox(input[rand_index[0]], lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

                # compute loss on mixed sample
                _, outputs = model(input)
                loss = criterion(outputs, labels_a)* lam + criterion(outputs, labels_b) * (1. - lam)
            else:
                # normal un-mixed sample
                _, outputs = model(input)
                loss = criterion(outputs, target)
            
            loss.backward()
            optimizer.step()

            decay_lr(optimizer, epoch=epoch, iteration=i)

            prediction = outputs.max(1)[1]
            correct = ( target == prediction ).sum().item()
            tepoch.set_postfix(loss=loss.item(), accuracy=100. * (correct/target.size(0)))
            i += 1
        i = 0

def evaluate(model, dataloader):
    total = 0
    correct = 0
    model.eval()

    for i, (input, target) in enumerate(dataloader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        _, outputs = model(input)
        prediction = outputs.max(1)[1]
        total += target.size(0)
        correct += ( target == prediction ).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy}")
    return accuracy

if __name__ == "__main__":
    wrn40_2 = wideresnet40_2().cuda()
    model_dict = { "wrn40_2": wrn40_2}
    results_dict = { "wrn40_2": 0 }
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        result = main(model, num_epochs=240, save_path=f"{model_name}_salmix.pth")
        results_dict[model_name] = result
    print(results_dict)
