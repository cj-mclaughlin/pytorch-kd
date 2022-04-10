"""
Baseline training procedure without distillation
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(1, os.path.abspath('..'))

from utils.dataset import get_cifar100_iterator
from utils.optimizer import get_default_optim, decay_lr
from models.resnet import ResNet, resnet20, resnet56, wideresnet16_2, wideresnet16_4, wideresnet28_4, wideresnet40_1, wideresnet40_2
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def main(model, num_epochs=240, save_path=""):
    optimizer = get_default_optim(model)
    train_dataloader, test_dataloader = get_cifar100_iterator()
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

if __name__ == "__main__":
    wrn40_2 = wideresnet40_2().cuda()
    wrn40_1 = wideresnet40_1().cuda()
    wrn28_4 = wideresnet28_4().cuda()
    wrn16_4 = wideresnet16_4().cuda()
    wrn16_2 = wideresnet16_2().cuda()
    resnet20 = resnet20().cuda()
    resnet56 = resnet56().cuda()
    model_dict = { "wrn40_2": wrn40_2, "wrn40_1": wrn40_1, "wrn28_4": wrn28_4, "wrn16_4": wrn16_4, "wrn16_2": wrn16_2, "resnet20": resnet20, "resnet56": resnet56 }
    results_dict = { "wrn40_2": 0, "wrn40_1": 0, "wrn28_4": 0, "wrn16_4": 0, "wrn16_2": 0, "resnet20": 0, "resnet56": 0 }
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        result = main(model, num_epochs=240, save_path=f"{model_name}.pth")
        results_dict[model_name] = result
    print(results_dict)
