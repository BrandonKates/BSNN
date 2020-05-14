from dataloaders import mnist_data
from models import gumbel_conv_lecun

import torch

device='cpu'
train, test, train_loader, test_loader = mnist_data.get(resize=True,batch_size=16)
model = gumbel_conv_lecun.GumbelConvLecunModel()
optimizer = torch.optim.Adam(model.parameters(), lr=.01)  
criterion = torch.nn.CrossEntropyLoss()
model.train()

for _, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        labels = labels.long()
        print('INPUT SHAPE', inputs.shape)
        optimizer.zero_grad()
        losses = [criterion(model(inputs), labels)]
        model.get_grad(losses)
        optimizer.step()
        break
