import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class XORDataset(Dataset):
    def __init__(self, n, d, sigma):
        self.inputs, self.labels = self.xorData(n, d, sigma)
        
    def xorData(self, n, d, sigma):
        mu_1 = [-1,-1]; mu_2 = [1,1]
        mu_3 = [-1, 1]; mu_4 = [1,-1]

        pos = np.random.normal(loc=mu_1, scale=sigma, size=(n,d))
        pos2 = np.random.normal(loc=mu_2, scale=sigma, size=(n,d))
        neg = np.random.normal(loc=mu_3, scale=sigma, size=(n,d))
        neg2 = np.random.normal(loc=mu_4, scale=sigma, size=(n,d))

        inputs = np.vstack((pos, pos2, neg, neg2))
        labels = np.concatenate((np.ones(2*n), np.zeros(2*n))) #label=0 for negative, label=1 for positive

        return np.float32(inputs), np.int_(labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx)
        print(self.labels)
        print(type(self.labels))
        print(self.labels.shape)
        print("INPUTS OUT: ", self.inputs[idx])
        print("LABELS OUT: ", self.labels[idx])
        return {self.inputs[idx], \
        self.labels[idx]}


def getDataLoader(n=100, d=2, sigma = 0.25, test_split = 0.2, batch_size = 1, num_workers = 1):
    train_n = int(n * (1-test_split))
    test_n = int(n * test_split)
    trainDataset = XORDataset(n=train_n, d=d, sigma=sigma)
    testDataset = XORDataset(n=test_n, d=d, sigma=sigma)
    
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return trainDataset, testDataset, trainDataLoader, testDataLoader