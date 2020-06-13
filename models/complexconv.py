import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import gumbel

class ComplexConv(nn.Module):
    def __init__(self, normalize, stochastic, device):
        super(ComplexConv, self).__init__()
        self.device = device
        self.stochastic = stochastic

        if self.stochastic:
            args = [device, normalize]

            self.conv1 = gumbel.Conv2d(3,64,3, *args)
            self.conv2 = gumbel.Conv2d(64,128,3, *args)
            self.conv3 = gumbel.Conv2d(128, 256, 3, *args)
            self.pool = nn.AvgPool2d(2, 2)
            self.fc1 = gumbel.Linear(64*4*4,128, *args)
            self.fc2 = gumbel.Linear(128, 256, *args)
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


    def forward(self, x):
        if self.stochastic:
            out = self.pool(self.conv1(x))
            out = self.pool(self.conv2(out))
            out = self.pool(self.conv3(out))

            out = out.view(-1, 64 * 4 * 4)
            return self.classifier(self.fc2(self.fc1(out)))
        else:
            out = self.pool(F.relu(self.conv1(x)))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(-1, 64 * 4 * 4)
            return self.classifier(F.relu(self.fc2(F.relu(self.fc1(out)))))


    def step(self):
        if self.stochastic:
            stoch_layers = [
                self.conv1, self.conv2, self.conv2, self.fc1, self.fc2
            ]

            for sl in stoch_layers:
                sl.step()
