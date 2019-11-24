from layers import bernoulli

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from math import pow
try:
    from listmodule import ListModule
except ImportError:
    from .listmodule import ListModule

class BernoulliModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(BernoulliModel, self).__init__()
        sizes = [input_size] + hidden_size_list + [output_size]
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(bernoulli.BernoulliLayer(sizes[i], sizes[i+1]))
        self.layers = ListModule(*self.layers)
        self.output_size = output_size

        
    def forward(self, x, with_grad=True):
        for layer in self.layers:
            x = layer(x, with_grad)
        return self.output_to_label(x)

    def get_grad(self, loss):
        for i in range(len(self.layers)):
            self.layers[i].get_grad(loss)
        
            
    def output_to_label(self, pred):
       ans = []
       for prediction in pred:
           dec = 0
           place = 1
           for neuron_out  in prediction:
               dec += (neuron_out + 1)/2 * place
               place *= 2
           ans.append(int(dec % int(pow(2,self.output_size))))
       return torch.from_numpy(np.array(ans))

   
    def predict(self, device):
        def func(x):
            print("PRED INPUT ", np.shape(x))
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            #Assume binary output
            ret = self.forward(x)
            print("PRED OUT ", len(ret))
            return ret
        return func
 
