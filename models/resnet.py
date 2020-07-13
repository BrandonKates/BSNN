import torch
import torch.nn as nn
import torchvision.models.resnet as deterministic_resnet

import layers as L
import sys

def conv3x3(in_planes, out_planes,device, stride=1, groups=1, dilation=1):
    return L.Conv2d(in_planes, out_planes, 3, device, True, stride=stride,
            padding=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, device, stride=1):
    return L.Conv2d(in_planes, out_planes, 1, device, True, stride=stride)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,device, stride=1, downsample=None,
            groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, device, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, device)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.ru1 = L.RU(planes, device)
        self.ru2 = L.RU(planes, device)


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ru1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        return self.ru2(out)


class ResNet(nn.Module):

    def __init__(self, block, layers, device, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.device = device
        self.in_planes = 64
        self.dilation = 1
        self.base_width = 64
        self.groups = groups
        self.conv1 = L.Conv2d(3, self.in_planes, 7, device, True, stride=2,
                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.ru = L.RU(self.in_planes, device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, 10)
        torch.nn.init.orthogonal_(self.fc.weight)
        #self.fc.weight.requires_grad = False


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes*block.expansion, self.device, 
                    stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, self.device, stride,
            downsample, self.groups, self.base_width, previous_dilation))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, self.device,
                groups=self.groups, base_width=self.base_width,
                dilation=self.dilation))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ru(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(stochastic, device):
    if stochastic:
        return ResNet(BasicBlock, [2,2,2,2], device)
    else:
        return deterministic_resnet.resnet18()

