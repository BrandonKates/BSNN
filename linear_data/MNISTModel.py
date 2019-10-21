import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from StochasticBinaryLayer import StochasticBinaryLayer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy
from numpy import ravel, reshape, swapaxes
from sklearn.metrics import confusion_matrix
from helpers import plot_confusion_matrix


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = StochasticBinaryLayer(784, 10)
        self.layer2 = StochasticBinaryLayer(28, 28)
        self.layer3 = StochasticBinaryLayer(28, 28)
        
    def forward(self, x, with_grad=True):
        x = x.view(-1, 28*28)
        x = self.layer1(x, with_grad)
        #x = self.layer2(x, with_grad)
        #x = self.layer3(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        #self.layer2.get_grad(loss)
        #self.layer3.get_grad(loss)
 
def run_model(num_epochs=100, learning_rate=0.001, train_loader = None, test_loader = None):
    model = MNISTModel().cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Move tensors to the configured device
            inputs = inputs.float().cuda()
            labels = labels.cuda()
            # Forward pass
            outputs = model(inputs)

            #loss = torch.sum((outputs - labels_onehot.cuda())**2) / batch_size
            #print("Inputs: ", inputs.shape)
            #print("Labels: ", labels.shape)
            #print("Outputs: ", outputs.shape)
            #print()
            loss = criterion(outputs, labels).cuda()
            # Backward and optimize
            model.get_grad(loss)
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Save the model checkpoint
    #print(model.state_dict())
    torch.save(model.state_dict(), 'models/mnist/MNIST_model.ckpt')
    print("Model saved to: ", os.getcwd() + "/models/mnist/MNIST_model.ckpt")

    # Test the model
    correct = 0
    total = 0
    conf_mat = np.zeros((10,10))
    for (inputs, labels) in test_loader:
        inputs = inputs.float().cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        conf_mat += confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

    print("Confusion Matrix:\n", conf_mat)
    print('Accuracy of the network on MNIST data: {} %'.format(100 * correct / total))

    plot_confusion_matrix(conf_mat, "Confusion Matrix MNIST Digits", 'MNIST')


if __name__ == "__main__":
    from load_MNIST import getMNISTDataLoader

    trainData, testData, train, test = getMNISTDataLoader()