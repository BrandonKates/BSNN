import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from listmodule import ListModule

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(LinearModel, self).__init__()
        sizes = [input_size] + hidden_size_list + [output_size]
        self.lin = []
        for i in range(len(sizes) - 1):
            self.lin.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
        self.lin = ListModule(*self.lin)
        
    def forward(self, x):
        for layer in self.lin:
            x = layer(x)
        return x

    def get_grad(self, loss):
        for layer in self.lin:
            layer.backward(loss)

    
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
