import torch
from torch import nn

import layers


class LeNet5(nn.Module):
    '''
    Simple model, almost exactly mimics lecun's architecture found here:
    http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf
    Just using this to test out stochastic convolutions
    '''
    def __init__(self, normalize, stochastic, device):
        super(LeNet5, self).__init__()
        self.stochastic = stochastic
        if stochastic:
            args = [device, normalize]

            # from linked paper top of page 4 and section 2.2
            module_list = [
                layers.Conv2d(1, 6, 5, *args),
                nn.AvgPool2d(2),
                layers.Conv2d(6, 16, 5, *args),
                nn.AvgPool2d(2),
                layers.Conv2d(16, 120, 5, *args),
                layers.Linear(120, 84, *args)
            ]
            self.linear_layer = nn.Linear(84, 10, bias=False)
            torch.nn.init.orthogonal_(self.linear_layer.weight)
            if stochastic:
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
        if self.stochastic:
            for i in filter(lambda i: i % 2 == 0, range(len(self.layers))):
                self.layers[i].step()
                
        else:
            return


    def forward(self, x):
        reshape_ind = 4 if self.stochastic else 9
        for layer_ind in range(len(self.layers)):
            x = self.layers[layer_ind](x)
            if layer_ind == reshape_ind:
                x = x.reshape(x.shape[0], x.shape[1])

        return self.linear_layer(x)
