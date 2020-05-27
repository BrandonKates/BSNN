import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import conv_layer

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super(BasicBlock, self).__init__()
        self.conv = conv_layer.Conv2dLayer(
                in_channels, 
                out_channels, 
                3, 
                padding=1, 
                bias=False,
                device=device)


    def forward(self, x, temp, with_grad=True):
        if with_grad:
            return self._forward(x, temp, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, temp, with_grad)


    def _forward(self, x, temp, with_grad):
        return torch.cat([x, self.conv(x, temp, with_grad)], 1)



class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super(TransitionBlock, self).__init__()
        self.conv = conv_layer.Conv2dLayer(
                in_channels, out_channels, 1, bias=False, device=device)


    def forward(self, x, temp, with_grad=True):
        if with_grad:
            return self._forward(x, temp, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, temp, with_grad)


    def _forward(self, x, temp, with_grad):
        out = self.conv(x, temp, with_grad)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, device='cpu'):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(in_channels, growth_rate, nb_layers, device)
        

    def _make_layer(self, in_channels, growth_rate, nb_layers, device):
        layers = []
        for i in range(nb_layers):
            layers.append(BasicBlock(in_channels+i*growth_rate, growth_rate, device))
        return nn.ModuleList(layers)


    def forward(self, x, temp, with_grad=True):
        if with_grad:
            return self._forward(x, temp, with_grad)
        else:
            with torch.no_grad():
                return self._forward(x, temp, with_grad)


    def _forward(x, temp, with_grad):
        for layer in self.layer:
            x = layer(x, temp, with_grad)
        return x


class DenseNet(nn.Module):
    def __init__(self, N=500, r=1e-5, device='cpu'):
        super(DenseNet, self).__init__()

        depth = 40
        growth_rate = 12
        num_classes = 10

        self.time_step = 0
        self.N = N
        self.r = r

        in_channels = 2*growth_rate
        n = int((depth - 4)/3)
        self.conv1 = conv_layer.Conv2dLayer(3, in_channels, 3,
                padding=1, bias=False, device=device)

        self.block1 = DenseBlock(n, in_channels, growth_rate, device=device)
        in_channels = int(in_channels+n*growth_rate)
        self.trans1 = TransitionBlock(in_channels, in_channels, device=device)

        self.block2 = DenseBlock(n, in_channels, growth_rate, device=device)
        in_channels = int(in_channels+n*growth_rate)
        self.trans2 = TransitionBlock(in_channels, in_channels, device=device)

        self.block3 = DenseBlock(n, in_channels, growth_rate, device=device)
        in_channels = int(in_channels+n*growth_rate)

        self.fc = nn.Linear(in_channels, 10, bias=False)
        nn.init.orthogonal_(self.fc.weight)
        self.fc.weight.requires_grad = False

        self.in_channels = in_channels

        for m in self.modules():
            if isinstance(m, conv_layer.Conv2dLayer):
                n = m.conv.kernel_size[0] * m.conv.kernel_size[1] * m.conv.out_channels
                m.conv.weight.data.normal_(0, math.sqrt(2. / n))


        def step(self):
            self.time_step += 1


        def _tau(self):
            return max(.5, exp(-self.r*floor(self.time_step/self.N)))


        def forward(self, x, with_grad=True):
            temp = self._tau()
            if with_grad:
                return self._forward(x, temp, with_grad)
            else:
                with torch.no_grad():
                    return self._forward(x, temp, with_grad)


        def _forward(self, x, temp, with_grad):
            out = self.conv1(x, temp, with_grad)
            out = self.trans1(self.block1(out, temp, with_grad), temp, with_grad)
            out = self.trans2(self.block2(out, temp, with_grad), temp, with_grad)
            out = self.block3(out, temp, with_grad)
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.in_channels)
            return self.fc(out)



