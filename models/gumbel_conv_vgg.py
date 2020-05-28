from math import exp, floor

from layers import gumbel, conv_layer
import layers

import numpy as np
import torch
from torch import nn

class GumbelConvVGGModel(nn.Module):
    '''
    TODO : stride for avgpool is none currently, vgg has 2. Also vgg has maxpool instead of avgpool
    Simple model, almost exactly mimics lecun's architecture found here:
    https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    Just using this to test out stochastic convolutions
    '''
    def __init__(self, device='cpu',N=500,r=1e-5,orthogonal=True):
        super(GumbelConvVGGModel, self).__init__()
        self.time_step = 0
        self.N = N
        self.r = r

        module_list = [
            conv_layer.Conv2dLayer(3, 64, 3, device=device),
            conv_layer.Conv2dLayer(64, 64, 3, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(64, 128, 3, device=device),
            conv_layer.Conv2dLayer(128, 512, 3, device=device, flatten=True),
            gumbel.GumbelLayer(51200, 4096, device=device),
            gumbel.GumbelLayer(4096, 4096, device=device)
        ]
        self.layers = nn.ModuleList(module_list)
        # paper uses radial basis function classification layer, I will use
        # what we normally use
        self.linear_layer = nn.Linear(4096, 10, bias=False)
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
            
'''
            conv_layer.Conv2dLayer(128, 128, 3, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(128, 256, 3, device=device),
            conv_layer.Conv2dLayer(256, 256, 3, device=device),
            conv_layer.Conv2dLayer(256, 256, 3, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(256, 512, 3, device=device),
            conv_layer.Conv2dLayer(512, 512, 3, device=device),
            conv_layer.Conv2dLayer(512, 512, 3, device=device),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(512, 512, 3, device=device),
            conv_layer.Conv2dLayer(512, 512, 3, device=device),
            conv_layer.Conv2dLayer(512, 512, 3, device=device, flatten=True),
            #nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(128, 512, 3, device=device, flatten=True),
            gumbel.GumbelLayer(512, 4096, device=device),
            gumbel.GumbelLayer(4096, 4096, device=device)
'''
