import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from StochasticBinaryLayer import StochasticBinaryLayer
import argparse


class StochasticBinaryModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(StochasticBinaryModel, self).__init__()
        self.layer1 = StochasticBinaryLayer(input_size, hidden_size)
        self.layer2 = StochasticBinaryLayer(hidden_size, num_classes)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        x = self.layer2(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        self.layer2.get_grad(loss)
    
    def predict(self,x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        #Apply softmax to output. 
        pred = F.softmax(self.forward(x), dim=0)
        ans = []
        #Pick the class with maximum weight
        for t in pred:
            if t[0]>t[1]:
                ans.append(-1)
            else:
                ans.append(1)
        return torch.tensor(ans)
 
def run_model(input_size = 2, hidden_size=3, num_classes=2, num_epochs=5, batch_size=1, learning_rate=0.001, train_loader = None, test_loader = None, device="cpu"):
    model = StochasticBinaryModel(input_size, hidden_size, num_classes).to(device)
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Move tensors to the configured device
            inputs = batch['input'].float().to(device)
            labels = batch['label']
            # Forward pass
            outputs = model(inputs)
            # One hot encoding buffer that you create out of the loop and just keep reusing
            labels_onehot = torch.FloatTensor(batch_size, num_classes)

            # In your for loop
            labels_onehot.zero_()
            labels_onehot.scatter_(1, (labels.long()).view(-1,1), 1)

            loss = torch.sum((outputs - labels_onehot.to(device))**2) / batch_size
            # Backward and optimize
            model.get_grad(loss)
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    correct = 0
    total = 0
    for batch in test_loader:
        inputs = batch['input'].float().to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on linearly separable data: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'models/model.ckpt')
    print("Model saved to: ", os.getcwd() + "/models/model.ckpt")

if __name__ == "__main__":
    '''
    from load_iris import getIrisDataLoader

    trainData, testData, train, test = getIrisDataLoader()

    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")

    from load_linearData import getLinearDataLoader

    # PARAMETERS:
    n = 100
    input_size = 2
    hidden_size = 1
    num_classes = 2
    num_epochs = 50
    batch_size = 1
    learning_rate = 0.001

    train_data, test_data, train, test = \
        getLinearDataLoader(n=n, d=num_classes, sigma = 0.15, test_split = 0.2, batch_size = 1, num_workers = 1)
    
    run_model(input_size = input_size, hidden_size=hidden_size, num_classes=num_classes, num_epochs=num_epochs,
        batch_size=batch_size, learning_rate=learning_rate,
        train_loader=train,
              test_loader=test, device=device)



