import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from math import pi, sin, cos

class CircleDataset(Dataset):
    def __init__(self, n, d, num_labels):
        self.data, self.targets = self.circleData(n, d, num_labels)
        
    def circleData(self, n, d, num_labels):
        center = [0 , 0]
        r = 1
        n //= num_labels
        labels = []
        inputs = []
        for label in range(num_labels):
            thetas = np.random.uniform(label * 2*pi / num_labels , (label + 1) * 2*pi / num_labels, n)
            labels += [label] * n
            inputs += [[r * cos(theta), r * sin(theta)] for theta in thetas]

        return np.float32(inputs), np.float32(labels)
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.targets[idx]


def get(n=100, d=2, num_labels=2, test_split = 0.2, batch_size = 1, num_workers = 1):
    train_n = int(n * (1-test_split))
    test_n = int(n * test_split)
    trainDataset = CircleDataset(n=train_n, d=d, num_labels=num_labels)
    testDataset = CircleDataset(n=test_n, d=d, num_labels=num_labels)

    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return trainDataset, testDataset, trainDataLoader, testDataLoader
