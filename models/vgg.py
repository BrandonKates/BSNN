import torch
import torch.nn as nn

from layers import conv_layer, gumbel

class VGG(nn.Module):
    def __init__(self, features, device='cpu',N=500,r=1e-5,stochastic=True):
        super(VGG, self).__init__()
        self.features = features
        self.time_step = 0
        self.N = N
        self.r = r
        self.stochastic = stochastic

        self.features = features
        if stochastic:
            pass
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512,512),
                nn.ReLU(True),
                nn.Linear(512,10),
            )
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2./n))
                    m.bias.data.zero_()


    def step(self):
        self.time_step += 1


    def tau(self):
        return max(.5, exp(-self.r*floor(self.time_step/self.N)))


    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x, with_grad)
        else:
            with torch.nograd():
                return self._forward(x, with_grad)


    def _forward(self, x, with_grad):
        if self.stochastic:
            pass
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)


def make_layers(cfg, stochastic, device, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            if stochastic:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            if stochastic:
                conv2d = conv_layer.Conv2dLayer(in_channels, v, 3, 
                        padding=2,device=device)
                if batch_norm:
                    pass # TODO
                else:
                    layers.append(conv2d)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg11(stochastic, device):
    return VGG(make_layers(cfg['A'], stochastic, device), 
            device=device, stochastic=stochastic)


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))
