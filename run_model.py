from os import getpid

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import confusion_matrix
#from psutil import Process

from optim import JangScheduler, ConstScheduler

def train(args, model, device, train_loader, optimizer, epoch, criterion, batch_size, temp_schedule=None):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step() 
        # adjust gumbel temperature 
        if temp_schedule:
            temp_schedule.step()
        if batch_idx % args.log_interval == 0:
            t = -math.inf if temp_schedule == None else temp_schedule.avg_temp()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTemp: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss.item(), t), flush=True)


def test(args, model, device, test_loader, criterion, batch_size, num_labels):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(args.inference_passes):
                outputs.append(model(inputs))
            mean_output = torch.mean(torch.stack(outputs), dim=0)
            pred = mean_output.argmax(dim=1)
            test_loss += criterion(mean_output, labels).sum().item()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )
    print("Confusion Matrix:\n", np.int_(conf_mat))


def _temp_scheduler(model, args):
    if args.temp_jang:
        N, r, limit = args.temp_step, args.temp_exp, args.temp_limit
        return JangScheduler(model.temperatures(), N, r, limit)
    else:
        return ConstScheduler(model.temperatures(), args.temp_const)


def run_model(model, args, criterion, train_loader, test_loader, num_labels, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
    temp_schedule = None if args.deterministic else _temp_scheduler(model, args)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion,
                args.batch_size, temp_schedule)
        test(args, model, device, test_loader, criterion, args.batch_size, num_labels)

    if not args.no_save:
        torch.save(model.state_dict(),
                f'checkpoints/{args.experiment_name}_{args.epoch}.pt')
