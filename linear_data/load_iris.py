import numpy as np
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn import datasets


class IrisDataset(Dataset):
    """Linearly Separable Dataset."""

    def __init__(self, dataset):
        self.inputs = dataset['inputs']
        self.labels = dataset['labels']
        self.length = len(self.labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        return self.inputs[idx], self.labels[idx]
    
def getDataLoader(test_split = 0.2, batch_size = 1, num_workers = 1):
    dataset = datasets.load_iris()
    inputs = np.float_(dataset.data)
    labels = dataset.target
    shuffle_dataset = True
    random_seed = 42

    # Creating data indices for training and validation splits:
    indices = list(range(len(labels)))
    split = int(np.floor(test_split * len(labels)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train = IrisDataset({'inputs': inputs[train_indices],
             'labels': labels[train_indices]})

    test = IrisDataset({'inputs': inputs[test_indices],
            'labels': labels[test_indices]})

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train, test, train_loader, test_loader