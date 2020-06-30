import torch
from torch import nn

import layers



class Linear(nn.Module):
    '''
    Extremely simple linear model to test multiple forward passes
    '''
    def __init__(self, normalize, stochastic, device):
        super(Linear, self).__init__()
        self.stochastic = stochastic
        if stochastic:
            args = [device, normalize]
            self.layers = nn.ModuleList([
                layers.Linear(1024, 300, *args),
                layers.Linear(300, 900, *args),
                layers.Linear(900, 300, *args),
            ])
            self.linear_layer = nn.Linear(300, 10, bias=False)
        else:
            self.layers = nn.ModuleList([
                nn.Linear(1024, 300), nn.Linear(300, 900), nn.Linear(900, 300),
            ])

        self.linear_layer = nn.Linear(300, 10, bias=False)
        if stochastic:
            torch.nn.init.orthogonal_(self.linear_layer.weight)
            self.linear_layer.weight.requires_grad = False


    def step(self):
        if self.stochastic:
            for layer in self.layers:
                layer.step()


    def forward(self, x, return_p=False):
        if return_p:
            x = x.view(-1, 1024)
            ps = []
            out = self.layers[0](x, True)
            ps.append(out[0])
            for layer in self.layers[1:]:
                out = layer(out[1], True)
                ps.append(out[0])
            ps = list(map(lambda x: torch.median(x).item(), ps))
            print(f"{ps[0], ps[1], ps[2]}")
            return self.linear_layer(out[1])
        else:
            x = x.view(-1, 1024)
            for layer in self.layers:
                x = layer(x)
            return self.linear_layer(x)

