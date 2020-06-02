from math import exp, floor

from layers import gumbel, conv_layer
import layers

import numpy as np
import torch
from torch import nn

class LeNet5(nn.Module):
    '''
    Simple model, almost exactly mimics lecun's architecture found here:
    http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf
    Just using this to test out stochastic convolutions
    '''
    def __init__(self,device='cpu',N=500,r=1e-5,orthogonal=True,stochastic=True):
        super(LeNet5, self).__init__()
        self.time_step = 0
        self.N = N
        self.r = r
        self.stochastic = stochastic
        if stochastic:
            # from linked paper top of page 4 and section 2.2
            module_list = [
                conv_layer.Conv2dLayer(1, 6, 5, device=device),
                nn.AvgPool2d(2),
                conv_layer.Conv2dLayer(6, 16, 5, device=device),
                nn.AvgPool2d(2),
                conv_layer.Conv2dLayer(16, 120, 5, device=device, flatten=True),
                gumbel.GumbelLayer(120, 84, device=device)
            ]
            self.linear_layer = nn.Linear(84, 10, bias=False)
            if orthogonal:
                torch.nn.init.orthogonal_(self.linear_layer.weight)
            self.linear_layer.weight.requires_grad = False

        else:
            module_list = [
                nn.Conv2d(1, 6, 5),
                nn.Tanh(),
                nn.AvgPool2d(2),
                nn.Tanh(),
                nn.Conv2d(6,16,5),
                nn.Tanh(),
                nn.AvgPool2d(2),
                nn.Tanh(),
                nn.Conv2d(16,120,5),
                nn.Tanh(),
                nn.Linear(120, 84),
            ]
            self.linear_layer = nn.Linear(84, 10, bias=False)

        self.layers = nn.ModuleList(module_list)


    def step(self):
        self.time_step += 1


    def tau(self):
        return max(.5, exp(-self.r*floor(self.time_step/self.N)))


    def _forward(self, x, with_grad):
        if self.stochastic:
            temp = self.tau()
            for layer in self.layers:
                if type(layer) == layers.conv_layer.Conv2dLayer:
                    x = layer(x, temp, with_grad)
                elif type(layer) == layers.gumbel.GumbelLayer:
                    x = layer(x, temp, with_grad)
                else:
                    x = layer(x)
        else:
            for layer_ind in range(len(self.layers)):
                x = self.layers[layer_ind](x)
                if layer_ind == 9:
                    x = x.reshape(x.shape[0], x.shape[1])

        return self.linear_layer(x)


    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, with_grad)
