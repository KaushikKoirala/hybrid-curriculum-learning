import torch.nn as nn
import torch
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def build_criterion(is_train=True):
    if is_train:
        return SoftTargetCrossEntropy()
    else:
        return nn.CrossEntropyLoss()