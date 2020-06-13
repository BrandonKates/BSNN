import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import gumbel

class SimpleConv(nn.Module):
    def __init__(self, normalize, stochastic, device):
        super(SimpleConv, self).__init__()
        self.device = device
        self.stochastic = stochastic

        if self.stochastic:
            args = [device, normalize]

            self.conv1 = gumbel.Conv2d(3,6,5, *args)
            self.conv2 = gumbel.Conv2d(6,16,5, *args)
            self.fc1 = gumbel.Linear(16*5*5,120, *args)
            self.fc2 = gumbel.Linear(120, 84, *args)
        else:
            self.conv1 = nn.Conv2d(3,6,5)
            self.conv2 = nn.Conv2d(6,16,5)
            self.fc1 = nn.Linear(16*5*5,120)
            self.fc2 = nn.Linear(120,84)

        self.classifier = nn.Linear(84, 10, bias=False)

        if self.stochastic:
            self.classifier.weight.requires_grad = False

        torch.nn.init.orthogonal_(self.classifier.weight)

    def forward(self, x):
        if self.stochastic:
            out = F.avg_pool2d(self.conv1(x), 2)
            out = F.avg_pool2d(self.conv2(out), 2)
            out = out.view(-1, 16 * 5 * 5)
            return self.classifier(self.fc2(self.fc1(out)))
        else:
            out = F.max_pool2d(F.relu(self.conv1(x)), 2)
            out = F.max_pool2d(F.relu(self.conv2(out)), 2)
            out = out.view(-1, 16 * 5 * 5)
            return self.classifier(F.relu(self.fc2(F.relu(self.fc1(out)))))


    def step(self):
        if self.stochastic:
            stoch_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
            for sl in stoch_layers:
                sl.step()
