from math import exp, floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import conv_layer, gumbel

class ComplexConv(nn.Module):
    def __init__(self, normalize, device='cpu',N=500,r=1e-5,stochastic=True):
        super(ComplexConv, self).__init__()
        self.device = device
        self.N = N
        self.r = r
        self.time_step = 0
        self.stochastic = stochastic

        if self.stochastic:
            kwargs = {'device': device, 'normalize': normalize}

            self.conv1 = conv_layer.Conv2dLayer(3,64,3, **kwargs)
            self.conv2 = conv_layer.Conv2dLayer(64,128,3, **kwargs)
            self.conv3 = conv_layer.Conv2dLayer(128, 256, 3, **kwargs)
            self.pool = nn.AvgPool2d(2, 2)
            self.fc1 = gumbel.GumbelLayer(64*4*4,128, **kwargs)
            self.fc2 = gumbel.GumbelLayer(128, 256, **kwargs)
        else:
            self.conv1 = nn.Conv2d(3,64, 3)
            self.conv2 = nn.Conv2d(64,128,3)
            self.conv3 = nn.Conv2d(128, 256, 3)
            self.pool = nn.MaxPool2d(2,2)
            self.fc1 = nn.Linear(64*4*4,128)
            self.fc2 = nn.Linear(128,256)

        self.classifier = nn.Linear(256, 10, bias=False)
        self.classifier.weight.requires_grad = False

        torch.nn.init.orthogonal_(self.classifier.weight)


    def print_grads(self):
        if self.stochastic:
            grads = [
                self.conv1.conv.weight.grad,
                self.conv2.conv.weight.grad,
                self.conv3.conv.weight.grad,
                self.fc1.lin.weight.grad,
                self.fc2.lin.weight.grad
            ]
        else:
            grads = [
                self.conv1.weight.grad,
                self.conv2.weight.grad,
                self.conv3.weight.grad,
                self.fc1.weight.grad,
                self.fc2.weight.grad
            ]

        print(list(map(lambda g: torch.norm(g).item(), grads)))


    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, with_grad)


    def _forward(self, x, with_grad):
        if self.stochastic:
            temp = self.tau()
            out = self.pool(self.conv1(x, temp, with_grad))
            out = self.pool(self.conv2(out, temp, with_grad))
            out = self.pool(self.conv3(out, temp, with_grad))

            out = out.view(-1, 64 * 4 * 4)
            return self.classifier(self.fc2(
                    self.fc1(out, temp, with_grad),
                temp, with_grad))
        else:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(-1, 64 * 4 * 4)
            return self.classifier(F.relu(self.fc2(F.relu(self.fc1(out)))))


    def step(self):
        self.time_step += 1


    def tau(self):
        return max(.5, exp(-self.r*floor(self.time_step/self.N)))



