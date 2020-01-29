import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from copy import deepcopy
import numpy as np
from torch import exp, log
 
class GumbelLayer(nn.Module):
    def __init__(self, input_dim, output_dim, new_loss_importance = 0.1, device="cpu"):
        super(GumbelLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        self.device = device

    def forward(self, x, with_grad):
        # print self.lin.dtype
        l = self.lin(x)
        # Change p to double so that gumbel_softmax func works
        p = torch.sigmoid(l).double()
        o = self.gumbel_softmax(p, 0.1)
        # Change output back to float for the next layer's input
        return o.float()

    def gumbel_softmax(self, p, temp):
        y1 = exp(( log(p) + self.sample_gumbel(p.shape) ) / temp)
        sum_all = y1 + exp(( log(1-p) + self.sample_gumbel(p.shape) ) / temp)
        return y1 / sum_all
        
        
    def sample_gumbel(self, input_size):
        u = torch.from_numpy(np.random.uniform(0, 1, input_size))
        return -log(-log(u))
        
    def parameters(self):
        # Everythin else is not trainable
        return self.lin.parameters()
    
'''    
    def predict(self,x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x), dim=0)
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(-1)
            else:
                ans.append(1)
        return torch.tensor(ans)
'''
