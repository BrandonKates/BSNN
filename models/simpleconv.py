from math import exp, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import conv_layer, gumbel

class SimpleConv(nn.Module):
    def __init__(self, device='cpu',N=500,r=1e-5,orthogonal=True,stochastic=True):
        super(SimpleConv, self).__init__()
        self.device = device
        self.N = N
        self.r = r
        self.time_step = 0
        self.stochastic = stochastic

        if self.stochastic:
            self.conv1 = conv_layer.Conv2dLayer(3,6,5, device=device)
            self.conv2 = conv_layer.Conv2dLayer(6,16,5, device=device)
            self.fc1 = gumbel.GumbelLayer(16*5*5,120, device=device)
            self.fc2 = gumbel.GumbelLayer(120, 84, device=device)
        else:
            self.conv1 = nn.Conv2d(3,6,5)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120)
            self.fc2 = nn.Linear(120,84)

        self.classifier = nn.Linear(84, 10, bias=False)

        if self.stochastic:
            self.classifier.weight.requires_grad = False

        if orthogonal:
            torch.nn.init.orthogonal_(self.classifier.weight)

    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, with_grad)


    def _forward(self, x, with_grad):
        if self.stochastic:
            temp = self.tau()
            out = F.avg_pool2d(self.conv1(x, temp, with_grad), 2)
            out = F.avg_pool2d(self.conv2(out, temp, with_grad), 2)
            out = out.view(-1, 16 * 5 * 5)
            return self.classifier(self.fc2(
                self.fc1(out, temp, with_grad),
                temp, with_grad))
        else:
            out = F.max_pool2d(F.relu(self.conv1(x)), 2)
            out = F.max_pool2d(F.relu(self.conv2(out)), 2)
            out = out.view(-1, 16 * 5 * 5)
            return self.classifier(F.relu(self.fc2(F.relu(self.fc1(out)))))


    def step(self):
        self.time_step += 1


    def tau(self):
        return max(.5, exp(-self.r*floor(self.time_step/self.N)))



