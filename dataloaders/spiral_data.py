import numpy as np
from numpy import pi
import torch 
from torch.utils.data import Dataset, DataLoader


class SpiralDataset(Dataset):
    def __init__(self, n):
        self.data, self.targets = self.spiral_data(n)

    def spiral_data(self, n):
        theta = np.sqrt(np.random.rand(n))*2*pi
        spiral_one = np.array([
            (np.cos(theta)*(2*theta+pi))/(2*pi),
            (np.sin(theta)*(2*theta+pi))/(2*pi)
        ]).T #+ np.random.randn(n, 2)
        spiral_two = np.array([
            (np.cos(theta)*(-2*theta-pi))/(2*pi),
            (np.sin(theta)*(-2*theta-pi))/(2*pi)
        ]).T #+ np.random.randn(n,2)
        
        data= np.vstack((spiral_one, spiral_two))
        labels = np.concatenate((np.zeros(n), np.ones(n)))

        return np.float32(data), np.float32(labels)
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.targets[idx]

def get(n=100, test_split=0.2, batch_size=1, num_workers=1):
    train_n = int(n * (1-test_split))
    test_n = int(n * test_split)
    trainDataset = SpiralDataset(n=train_n)
    testDataset = SpiralDataset(n=test_n)
    
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testDataLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return trainDataset, testDataset, trainDataLoader, testDataLoader
