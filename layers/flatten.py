import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from copy import deepcopy
from torch import exp, log
 
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x, temp, with_grad):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
