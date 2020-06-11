import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from copy import deepcopy
from torch import exp, log
 
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True, flatten=False, device="cpu", normalize=True):
        super(Conv2dLayer, self).__init__()
        self.conv   = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, 
                padding=padding, bias=bias)
        self.device = device
        self.flatten = flatten
        self.normalize = normalize
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x, temp, with_grad, debug=False):
        l = self.conv(x)
        if self.normalize:
            l = self.batchnorm(l)
        if self.flatten:
            l = l.view(l.shape[0], -1)
            #l = l.reshape((l.shape[0], l.shape[1]))
        # Change p to double so that gumbel_softmax func works
        delta = 1e-5
        p = torch.clamp(torch.sigmoid(l).double(), min=delta, max=1-delta)
        if debug:
            print(p)
        o = self.gumbel_softmax(p, temp)
        # Change output back to float for the next layer's input
        return o.float()


    def gumbel_softmax(self, p, temp):
        y1 = exp(( log(p) + self.sample_gumbel(p.shape) ) / temp)
        sum_all = y1 + exp(( log(1-p) + self.sample_gumbel(p.shape) ) / temp)
        return y1 / sum_all
        
        
    def sample_gumbel(self, input_size):
        #u = torch.rand(input_size).to(self.device)
        if self.device == torch.device('cpu'):
            u = torch.FloatTensor(input_size).uniform_()
        else:
            u = torch.cuda.FloatTensor(input_size).uniform_()
        return -log(-log(u))
