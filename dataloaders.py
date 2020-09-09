import logging
import json
import os
import random

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10, MNIST, SVHN, STL10
from torchvision import transforms


SPLIT_FILE = 'splits.json'

def _get_train_val_samplers(dataset, trainset, split=[.9,.1]):
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, 'r') as fp:
            indexes = json.load(fp)
            if dataset in indexes.keys():
                s = indexes[dataset]
                val_inds, train_inds = s['val'], s['train']
                if len(val_inds)/len(trainset) == split[1]:
                    train_sampler = SubsetRandomSampler(train_inds)
                    val_sampler = SubsetRandomSampler(val_inds)
                    return train_sampler, val_sampler
                else:
                    logging.info('Recomputing train and val indexes')

    else:
        indexes = {}

    num_train_samples = len(trainset)
    indices = list(range(num_train_samples))
    random.shuffle(indices)
    num_val = int((split[1]*num_train_samples))
    val_inds, train_inds = indices[:num_val], indices[num_val:]
    indexes[dataset] = {'val': val_inds, 'train': train_inds}
    with open(SPLIT_FILE, 'w') as fp:
        json.dump(indexes, fp)
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)

    return train_sampler, val_sampler


def mnist(resize=False, test_split=0.2, batch_size=1, num_workers=1):
    mnist_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    if resize:
        mnist_transforms = [transforms.Resize(32)] + mnist_transforms
    mnist_transforms = transforms.Compose(mnist_transforms)

    mnist_train = MNIST('MNIST_DATA/', train=True, transform=mnist_transforms, download=True)
    mnist_test = MNIST('MNIST_DATA/', train = False,transform=mnist_transforms)
    train_sampler, val_sampler = _get_train_val_samplers('mnist', mnist_train)

    train_loader = DataLoader(mnist_train, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(mnist_train, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers)
    test_loader  = DataLoader(mnist_test,  batch_size=1000, shuffle=True,
            num_workers=num_workers)

    return train_loader, val_loader, test_loader


def cifar10(batch_size, num_workers=1):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261]
    )

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = CIFAR10(root='./CIFAR10_DATA', train=True, download=True,
                        transform=train_transform)
    testset = CIFAR10(root='./CIFAR10_DATA', train=False, download=True,
                        transform=test_transform)

    train_sampler, val_sampler = _get_train_val_samplers('cifar10', trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             num_workers=num_workers, sampler=train_sampler)
    valloader = DataLoader(trainset, batch_size=batch_size, 
                            num_workers=num_workers, sampler=val_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)

    return trainloader, valloader, testloader


def svhn(batch_size, num_workers=1):
    ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.4377, .4438, .4782],
                          std=[.1282, .1315, .1123])])
    trainset = SVHN('./SVHN', transform=ts, download=True)
    testset = SVHN('./SVHN', split='test', download=True, transform=ts)
    train_sampler, val_sampler = _get_train_val_samplers('svhn', trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
    valloader = DataLoader(trainset, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    return trainloader, valloader, testloader
