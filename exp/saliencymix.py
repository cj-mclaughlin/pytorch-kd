"""
SaliencyMix Training https://arxiv.org/abs/2006.01791
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from utils.dataset import baseline_cifar100_dataloader
from utils.optimizer import get_default_optim, decay_lr
from models.resnet import *
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

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
    criterion = CrossEntropyLoss()
    
    i = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for input, target in tepoch:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
            optimizer.zero_grad()
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