from os import getpid

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from psutil import Process


def cpu_stats():
    pid = getpid()
    py = Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


def train(args, model, device, train_loader, optimizer, epoch, criterion, batch_size):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
#        model.print_grad()
        optimizer.step() 
        # adjust gumbel temperature 
        model.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTemp: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss.item(), model.tau()))


def test(args, model, device, test_loader, criterion, batch_size, num_labels):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        passes_pred = []
        output = model(inputs, with_grad=False)
        test_loss += criterion(output, labels).sum().item() # sum up batch loss
        passes_pred.append(output.argmax(dim=1, keepdim=True))
        pred = torch.mode(torch.cat(passes_pred, dim=1), dim=1, keepdim=True)[0]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )
    print("Confusion Matrix:\n", np.int_(conf_mat))



def run_model(model, args, criterion, train_loader, test_loader, num_labels, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=.9, weight_decay=5e-4)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion, args.batch_size)
        test(args, model, device, test_loader, criterion, args.batch_size, num_labels)
        if epoch % 50 == 0:
            if (args.save_model):
                torch.save(model.state_dict(), args.save_location + str(epoch))
