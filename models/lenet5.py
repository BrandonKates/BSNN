import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self, orthogonal=True):
        super(LeNet5, self).__init__()
        moduleList = [
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Tanh(),
            nn.Conv2d(6,16,5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Tanh(),
            nn.Conv2d(16,120,5),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Linear(84, 10, bias=False)
        ]

        self.layers = nn.ModuleList(moduleList)
        if orthogonal:
            nn.init.orthogonal_(moduleList[-1].weight)


    def _forward(self, x):
        for layer_ind in range(len(self.layers)):
            x = self.layers[layer_ind](x)
            if layer_ind == 9:
                x = x.reshape(x.shape[0], x.shape[1])
        return x


    def forward(self, x, with_grad=True):
        if with_grad:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


    def get_grad(self, losses):
        for loss in losses:
            loss.backward()
