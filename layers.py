import math

import torch
from torch import nn
from torch import exp, log

from optim import TempVar
 
class _GumbelLayer(nn.Module):
    ''' 
    Base class for all gumbel-stochastic layers
    '''
    def __init__(self, inner, device, norm):
        '''
        inner:
            pytorch module subclass instance used as deterministic inner class
        device: 
            cpu or gpu. Needed for gumbel sample, would be nice to make
            obselete
        norm:
           batch norm object
        '''
        super(_GumbelLayer, self).__init__()
        self.inner = inner
        self.device = device
        if norm:
            self.norm = norm
        else:
            self.norm = nn.Identity()

        self.temp = TempVar()


    def forward(self, x):
        l = self.inner(x)
        l = self.norm(l)
        # Change p to double so that gumbel_softmax func works
        delta = 1e-5
        p = torch.clamp(torch.sigmoid(l).double(), min=delta, max=1-delta)
        o = self.sample(p)
        # Change output back to float for the next layer's input
        return o.float()


    def sample(self, p):
        if self.training:
            # sample relaxed bernoulli dist
            return self._gumbel_softmax(p) 
        else:
            return torch.bernoulli(p).to(self.device)


    def _gumbel_softmax(self, p):
        y1 = exp(( log(p) + self._sample_gumbel_dist(p.shape) ) / self.temp.val)
        sum_all = y1 + exp(( log(1-p) + self._sample_gumbel_dist(p.shape))
                / self.temp.val)
        return y1 / sum_all
        
        
    def _sample_gumbel_dist(self, input_size):
        if self.device == torch.device('cpu'):
            u = torch.FloatTensor(input_size).uniform_()
        else:
            u = torch.cuda.FloatTensor(input_size).uniform_()
        return -log(-log(u))


class Linear(_GumbelLayer):
    def __init__(self, input_dim, output_dim, device, norm):
        inner = nn.Linear(input_dim, output_dim, bias=False)
        norm_obj = nn.BatchNorm1d(output_dim) if norm else None
        super(Linear, self).__init__(inner, device, norm_obj)


class Conv2d(_GumbelLayer):
    def __init__(self, inc, outc, kernel, device, norm, **kwargs):
        inner = nn.Conv2d(inc, outc, kernel, **kwargs)
        norm_obj = nn.BatchNorm2d(outc) if norm else None
        super(Conv2d, self).__init__(inner, device, norm_obj)


class RU(_GumbelLayer):
    def __init__(self, output_dim , device):
        inner = nn.Identity()
        norm_obj = nn.BatchNorm2d(output_dim)
        super(RU, self).__init__(inner, device, norm_obj)

