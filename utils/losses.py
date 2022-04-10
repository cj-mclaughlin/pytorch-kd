import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    """
    Temperature-Scaled Softmax Target for Distillation

    Distilling the Knowledge in a Neural Network
        https://arxiv.org/pdf/1503.02531.pdf

    Implementation from https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/KD.py
    Common values in literature for T are 3.0 and 4.0. 
    """
    def __init__(self, T=4.0):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss