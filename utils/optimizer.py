from torch.optim import SGD

LR = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_DECAY = 0.1
STEP_EPOCHS = [150, 180, 210]

def get_default_optim(model):
    optim = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    return optim

def decay_lr(optim, epoch, iteration, method="stepwise"):
    assert method in ["stepwise"]
    return stepwise_decay(optim, epoch, iteration)

def stepwise_decay(optim, epoch, iteration):
    if epoch in STEP_EPOCHS and iteration == 0:
        print("Decaying Learning Rate!")
        for group in optim.param_groups:
            group['lr'] = group['lr'] * STEP_DECAY
    
        