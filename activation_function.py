from torch import nn as nn
from torch.nn import functional as F
import torch
def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

class Mish(nn.Module):
    """paper:A self Regularized Non-Monotonic Neural Activation Function"""
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x*(torch.tanh(F.softplus(x)))
        return x
class Selu(nn.Module):
    def __init__(self):
        super().__init__()
        print("Selu activation loaded...")
    def forward(self,x):
        return F.selu(x,inplace=True)
