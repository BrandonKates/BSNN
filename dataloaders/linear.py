import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class LinearDataset(Dataset):
    """Linearly Separable Dataset."""

    def __init__(self, n, d, sigma = 0.15):
        self.inputs, self.labels = self.linearData(n, d, sigma)
        
    def linearData(self, n, d, sigma):
        mu_1 = -0.5; mu_2 = 0.5

        pos = np.random.normal(loc=mu_1, scale=sigma, size=(n,d))
        neg = np.random.normal(loc=mu_2, scale=sigma, size=(n,d))

        inputs = np.vstack((pos, neg))
        labels = np.concatenate((np.ones(n), np.zeros(n))) #negative examples have label == 0 => True

        return np.float32(inputs), np.int_(labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {self.inputs[idx], self.labels[idx]}

def get(n=100, d=2, sigma = 0.15, test_split = 0.2, batch_size = 1, num_workers = 1):
    train_n = int(n * (1-test_split))
    test_n = int(n * test_split)
    trainDataset = LinearDataset(n=train_n, d=d, sigma=sigma)
    testDataset = LinearDataset(n=test_n, d=d, sigma=sigma)
    
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return trainDataset, testDataset, trainDataLoader, testDataLoader
