from math import exp, floor

from layers import gumbel, conv_layer
import layers

import numpy as np
import torch
from torch import nn

class GumbelConvLecunModel(nn.Module):
    '''
    Simple model, almost exactly mimics lecun's architecture found here:
    http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf
    Just using this to test out stochastic convolutions
    '''
    def __init__(self, device='cpu',N=500,r=1e-5,orthogonal=True):
        super(GumbelConvLecunModel, self).__init__()
        self.time_step = 0
        self.N = N
        self.r = r
        # from linked paper top of page 4 and section 2.2
        module_list = [
            conv_layer.Conv2dLayer(1, 6, 5, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(6, 16, 5, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(16, 120, 5, device=device, flatten=True),
            gumbel.GumbelLayer(120, 84, device=device)
        ]
        self.layers = nn.ModuleList(module_list)
        # paper uses radial basis function classification layer, I will use
        # what we normally use
        self.linear_layer = nn.Linear(84, 10, bias=False)
        if orthogonal:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
        self.linear_layer.weight.requires_grad = False


    def step(self):
        self.time_step += 1


    def _tau(self):
        return max(.5, exp(-self.r*floor(self.time_step/self.N)))


    def _forward(self, x, with_grad):
        temp = self._tau()
        for layer in self.layers:
            if type(layer) == layers.conv_layer.Conv2dLayer:
                x = layer(x, temp, with_grad)
            elif type(layer) == layers.gumbel.GumbelLayer:
                x = layer(x, temp, with_grad)
            else:
                x = layer(x)
        return self.linear_layer(x)


    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, with_grad)
