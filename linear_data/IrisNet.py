from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from StochasticBinaryLayer import StochasticBinaryLayer
from load_iris import getIrisDataLoader

import numpy as np
from sklearn.metrics import confusion_matrix
from helpers import plot_confusion_matrix


class IrisNet(nn.Module):
    def __init__(self, device):
        super(IrisNet, self).__init__()
        self.layer1 = StochasticBinaryLayer(4, 3, device=device)
        #self.layer2 = StochasticBinaryLayer(10, 3, device=device)
        #self.layer3 = StochasticBinaryLayer(16, 3, device=device)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        #x = self.layer2(x, with_grad)
        #x = self.layer3(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        #self.layer2.get_grad(loss)
        #self.layer3.get_grad(loss)
    
    def predict(self,x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        #Apply softmax to output.
        pred = F.softmax(self.forward(x), dim=1)

        ans = []
        for prediction in pred:
            ans.append(prediction.argmax().item())
        return ans
    
def train(args, model, device, train_loader, optimizer, epoch, criterion):
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

def test(args, model, device, test_loader, criterion):
    conf_mat = np.zeros((3,3))
    model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        output = model(inputs)
        test_loss += criterion(output, labels).sum().item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

        #print(labels, pred)
        #conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())


    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #print("Confusion Matrix:\n", np.int_(conf_mat))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    _,_,train_loader, test_loader = getIrisDataLoader()

    model = IrisNet(device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        test(args, model, device, test_loader, criterion)

    if (args.save_model):
        torch.save(model.state_dict(),'models/iris/iris_BSNN.pt')
        
if __name__ == '__main__':
    main()