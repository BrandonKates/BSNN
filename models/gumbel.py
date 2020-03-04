from layers import gumbel

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from math import pow

class GumbelModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, num_labels, device='cpu', orthogonal=True):
        super(GumbelModel, self).__init__()
        sizes = [input_size] + hidden_size_list
        self.layers = nn.ModuleList([gumbel.GumbelLayer(sizes[i], sizes[i+1], device=device) for i in range(len(sizes)-1)])
        self.linear_layer = nn.Linear(sizes[-1], output_size, bias=False)
        if orthogonal:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
            self.linear_layer.weight.requires_grad = False
        self.num_labels = num_labels
        self.device = device

    def forward(self, x, with_grad):
        for layer in self.layers:
            x = layer(x, with_grad)
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
 
