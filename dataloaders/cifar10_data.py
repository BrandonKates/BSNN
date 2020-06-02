import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10 
from torchvision import transforms

import sys

def get(batch_size, num_workers=1):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
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
