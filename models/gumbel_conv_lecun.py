from math import exp, floor

from layers import gumbel, conv_layer

import numpy as np
import torch
from torch import nn

class GumbelConvLecunModel(nn.Module):
    '''
    Simple model, almost exactly mimics lecun's architecture found here:
    http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf
    Just using this to test out stochastic convolutions
    '''
    def __init__(self, device='cpu', orthogonal=True):
        super(GumbelConvLecunModel, self).__init__()
        # from linked paper top of page 4 and section 2.2
        module_list = [
            conv_layer.Conv2dLayer(1, 6, 5),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(6, 16, 5),
            nn.AvgPool2d(2),
            conv_layer.Conv2dLayer(16, 120, 5, flatten=True),
            nn.Linear(120, 84) # TODO replace with linear Gumbel layer
        ]
        self.layers = nn.ModuleList(module_list)
        # paper uses radial basis function classification layer, I will use
        # what we normally use
        self.linear_layer = nn.Linear(84, 10, bias=False)
        if orthogonal:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
        self.linear_layer.weight.requires_grad = False
        # TODO temperature


    def forward(self, x, with_grad=True):
        x = self.layers[0].forward(x, .5, with_grad)
        x = self.layers[1].forward(x)
        x = self.layers[2].forward(x, .5, with_grad)
        x = self.layers[3].forward(x)
        x = self.layers[4].forward(x, .5, with_grad)
        x = self.layers[5](x)
        x = self.linear_layer(x)
        return x


    def get_grad(self, losses):
        for loss in losses:
            loss.backward()

