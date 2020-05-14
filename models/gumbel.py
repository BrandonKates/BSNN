from math import exp, floor

from layers import gumbel, conv_layer, flatten

import torch
from torch import nn
import numpy as np

STEP = 1000
RATE = 1e-5

class GumbelModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, num_labels, temp, device='cpu', orthogonal=True):
        super(GumbelModel, self).__init__()
        sizes = [input_size] + hidden_size_list
        module_list = []
        for i in range(len(sizes)-1):
            module_list.append(conv_layer.Conv2dLayer(sizes[i], sizes[i+1], device=device))
        module_list.append(flatten.Flatten())
        self.layers = nn.ModuleList(module_list)
        self.linear_layer = nn.Linear(32*32*hidden_size_list[-1], output_size, bias=False)
        if orthogonal:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
            self.linear_layer.weight.requires_grad = False
        self.num_labels = num_labels
        self.device = device
        self.time_step = 0

        if temp == 'schedule':
            self.tau = lambda : max(.5, 2*exp(-1*RATE*floor(self.time_step/STEP)))
        else:
            temp = float(temp)
            self.tau = lambda: temp


    def step(self):
        self.time_step += 1

    def forward(self, x, with_grad=True):
        for layer in self.layers:
            x = layer(x, self.tau(), with_grad)
        return self.linear_layer(x)


    def get_grad(self, losses):
        for loss in losses:
            loss.backward()

    def predict(self, device, num_passes):
        def func(x):
            print("PRED INPUT ", np.shape(x))
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            passes_pred = []
            for i in range(num_passes):
                output = self.forward(x, with_grad=False)
                passes_pred.append(output.argmax(dim=1, keepdim=True))
            pred = torch.mode(torch.cat(passes_pred, dim=1), dim=1, keepdim=True)[0]
            return pred
        return func
 
