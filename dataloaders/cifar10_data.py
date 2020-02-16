import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10 
from torchvision import transforms

import sys

def get(batch_size, num_workers=1):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # is this necessary?
    ])

    trainset = CIFAR10(root='./CIFAR10_DATA', 
            train=True, download=True,
            transform=transform )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)

    testset = CIFAR10(root='./CIFAR10_DATA', 
            train=False, download=True,
            transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return trainset, testset, trainloader, testloader
