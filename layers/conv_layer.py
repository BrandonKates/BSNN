import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from copy import deepcopy
from torch import exp, log
 
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, flatten=False, device="cpu"):
        super(Conv2dLayer, self).__init__()
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.device = device
        self.flatten = flatten


    def forward(self, x, temp, with_grad):
        l = self.conv(x)
        if self.flatten:
            l = l.reshape((l.shape[0], l.shape[1]))
        # Change p to double so that gumbel_softmax func works
        delta = 1e-5
        p = torch.clamp(torch.sigmoid(l).double(), min=delta, max=1-delta)
        o = self.gumbel_softmax(p, temp)
        # Change output back to float for the next layer's input
        return o.float()


    def gumbel_softmax(self, p, temp):
        y1 = exp(( log(p) + self.sample_gumbel(p.shape) ) / temp)
        sum_all = y1 + exp(( log(1-p) + self.sample_gumbel(p.shape) ) / temp)
        return y1 / sum_all
        
        
    def sample_gumbel(self, input_size):
        u = torch.rand(input_size).to(self.device)
        return -log(-log(u))
