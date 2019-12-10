from layers import bernoulli

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from math import pow

class BernoulliModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, num_labels, device='cpu', orthogonal=True):
        super(BernoulliModel, self).__init__()
        sizes = [input_size] + hidden_size_list
        self.layers = nn.ModuleList([bernoulli.BernoulliLayer(sizes[i], sizes[i+1], device=device) for i in range(len(sizes)-1)])
        self.linear_layer = nn.Linear(sizes[-1], output_size)
        if orthogonal:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
        self.num_labels = num_labels
        self.device = device

    def forward(self, x, with_grad=True):
        for layer in self.layers:
            x = layer(x, with_grad)
        return self.linear_layer(x)

    def get_grad(self, loss):
        for i in range(len(self.layers)):
            self.layers[i].get_grad(loss)

    def predict(self, device):
        def func(x):
            print("PRED INPUT ", np.shape(x))
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            pred = F.softmax(self.forward(x), dim=1)
            ans = []
            for prediction in pred:
                ans.append(prediction.argmax().item())
            return ans
        return func
 
