import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.metrics import confusion_matrix

from helpers import plot_confusion_matrix, plot_decision_boundary
from layers import bernoulli

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device='cpu'):
        super(Net, self).__init__()
        self.layer1 = bernoulli.BernoulliLayer(input_size, hidden_size, device=device)
        self.layer2 = bernoulli.BernoulliLayer(hidden_size, num_classes, device=device)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        x = self.layer2(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        self.layer2.get_grad(loss)
    
    def predict(self, device):
        def func(x):
            x = torch.from_numpy(x).type(torch.FloatTensor).to(device)
            #Apply softmax to output.
            pred = F.softmax(self.forward(x), dim=1)

            ans = []
            for prediction in pred:
                ans.append(prediction.argmax().item())
            return ans
        return func


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
        pred = output.squeeze()#output.argmax(dim=1, keepdim=True).float() # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item() #torch.all(output.eq(labels)).sum().item()
        #correct += output.eq(labels).sum(1).eq(num_classes).sum().item()
        #conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())


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


