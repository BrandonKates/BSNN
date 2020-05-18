import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
try:
    from listmodule import ListModule
except ImportError:
    from .listmodule import ListModule

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size_list, output_size):
        super(LinearModel, self).__init__()
        sizes = [input_size] + hidden_size_list + [output_size]
        self.lin = []
        for i in range(len(sizes) - 1):
            self.lin.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
        self.lin = ListModule(*self.lin)
        
    def forward(self, x, with_grad=True):
        if not with_grad:
            with torch.no_grad():
                for layer in self.lin:
                    x = layer(x)
        else:
            for layer in self.lin:
                x = layer(x)

        return x

    def get_grad(self, losses):
        for loss in losses:
            loss.backward()

    
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
