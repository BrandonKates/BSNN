import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self, device='cpu'):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(2,8, bias=True)
        self.lin2 = nn.Linear(8,2, bias=True)
        
    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        return x
    
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