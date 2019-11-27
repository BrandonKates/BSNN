from layers import bernoulli

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from math import pow

class BernoulliModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size, num_labels, device='cpu'):
        super(BernoulliModel, self).__init__()
        sizes = [input_size] + hidden_size_list + [output_size]
        self.layers = nn.Sequential(*[bernoulli.BernoulliLayer(sizes[i], sizes[i+1], device=device) for i in range(len(sizes)-1)])
        self.output_size = output_size
        self.num_labels = num_labels
        self.device = device
        self.place = torch.IntTensor([2**i for i in range(self.output_size)]).to(device)

    def forward(self, x, with_grad=True):
        for layer in self.layers:
            x = layer(x, with_grad)
        return self.output_to_label(x)

    def get_grad(self, loss):
        for i in range(len(self.layers)):
            self.layers[i].get_grad(loss)

    # torch version
    def output_to_label(self, pred):
      return ((pred.int() + 1) / 2 * self.place).sum(dim=1) % 10

    def predict(self, device):
        def func(x):
            print("PRED INPUT ", np.shape(x))
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            #Assume binary output
            ret = self.forward(x)
            print("PRED OUT ", len(ret))
            return ret
        return func
 
