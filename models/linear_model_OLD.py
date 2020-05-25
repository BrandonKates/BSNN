import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.metrics import confusion_matrix

from helpers import plot_confusion_matrix, plot_decision_boundary
from layers import bernoulli

def train(args, model, device, train_loader, optimizer, epoch, criterion, batch_size):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        model.get_grad(loss)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, criterion, batch_size, num_classes):
    conf_mat = np.zeros((2, 2))
    model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        output = model(inputs)
        print(output)
        print(labels)
        test_loss += criterion(output, labels).sum().item() # sum up batch loss
        pred = torch.max(outputs.data, 1)
        correct += pred.eq(labels.view_as(pred)).sum().item() #torch.all(output.eq(labels)).sum().item()
        #correct += output.eq(labels).sum(1).eq(num_classes).sum().item()
        conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())

        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print("Confusion Matrix:\n", np.int_(conf_mat))

def run_model(args, criterion, train_loader, test_loader, device, input_size, hidden_size, num_classes):
    model = Net(input_size, hidden_size, num_classes, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion, args.batch_size)
        test(args, model, device, test_loader, criterion, args.batch_size, num_classes=num_classes)

    if (args.save_model):
        torch.save(model.state_dict(), args.save_location)

    plot_decision_boundary(model.predict(device), test_loader.dataset.inputs, test_loader.dataset.labels, save_name='xor_bernoulli')


