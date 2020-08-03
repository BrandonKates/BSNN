import math

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.models.

import layers as L

class _BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, device):
        super(_BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = L.Conv2d(in_planes, out_planes, 3, device, True, stride=1,
                padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.bn(x))
        return torch.cat([x, out], 1)



class _BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, device):
        super(_BottleneckBlock, self,).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = L.Conv2d(in_planes, inter_planes, 1, device, True,
                stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = L.Conv2d(inter_planes, out_planes, 3, device, True,
                stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(self.bn2(out))
        return torch.cat([x, out], 1)



class _TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, device):
        super(_TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = L.Conv2d(in_planes, out_planes, 1, device, True, stride=1,
                padding=0, bias=False)


    def forward(self, x):
        return F.avg_pool2d(self.conv(self.bn(x)), 2)


class _DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, device):
        super(_DenseBlock, self).__init__()
        self.device = device
        self.block = block
        self.layer = self._make_layer(in_planes, growth_rate, nb_layers)

    def _make_layer(self, in_planes, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(self.block(in_planes+i*growth_rate, growth_rate,
                self.device))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class _DenseNet(nn.Module):
    def __init__(self, depth, device, growth_rate=12,
            reduction=.5,bottleneck=True):
        super(_DenseNet, self).__init__()
        in_planes = 2*growth_rate
        n = (depth - 4)/3
        if bottleneck == True:
            n = n/2
            block = _BottleneckBlock
        else:
            block = _BasicBlock

        n = int(n)
        self.conv1 = L.Conv2d(3, in_planes, 3, device, True, stride=1,
                padding=1, bias=False)

        self.block1 = _DenseBlock(n, in_planes, growth_rate, block, device)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = _TransitionBlock(in_planes,
                int(math.floor(in_planes*reduction)), device)
        in_planes = int(in_planes*reduction)

        self.block2 = _DenseBlock(n, in_planes, growth_rate, block, device)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = _TransitionBlock(in_planes,
                int(math.floor(in_planes*reduction)), device)
        in_planes = int(in_planes * reduction)

        self.block3 = _DenseBlock(n, in_planes, growth_rate, block, device)
        in_planes = int(in_planes+n*growth_rate)

        self.bn1 = nn.BatchNorm2d(in_planes)

        self.fc = nn.Linear(in_planes, 10)
        torch.nn.init.orthogonal_(self.fc.weight)
        self.fc.weight.requires_grad = False
        self.in_planes = in_planes

        # init weights
        for m in self.modules():
            if isinstance(m, L.Conv2d):
                m = m.inner
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.bn1(self.block3(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


    def temperatures(self):
        temps = [self.conv1.temp]
        for m in self.modules():
            if isinstance(m, L.Conv2d):
                temps.append(m.temp)
        return temps


def densenet(stochastic, depth, device, growth_rate, reduction, bottleneck):
    if stochastic:
        return _DenseNet(depth, device, growth_rate, reduction, bottleneck)
    else:
        pass # TODO

