from layers import bernoulli

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse


class BernoulliModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(BernoulliModel, self).__init__()
        sizes = [input_size] + hidden_size_list
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(bernoulli.BernoulliLayer(sizes[i], sizes[i+1]))
        self.layers.append(nn.Linear(sizes[-1], output_size))
        
        
    def forward(self, x):
        for layer in self.lin:
            x = layer(x)
        return x

    def get_grad(self, loss):
        for layer in self.layers:
            layer.get_grad(loss)
    
    def predict(self, device):
        def func(x):
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            #Apply softmax to output.
            pred = F.softmax(self.forward(x), dim=1)

            ans = []
            for prediction in pred:
                ans.append(prediction.argmax().item())
            return ans
        return func
 
