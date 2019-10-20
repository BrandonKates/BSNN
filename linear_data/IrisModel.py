import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from StochasticBinaryLayer import StochasticBinaryLayer

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.layer1 = StochasticBinaryLayer(4, 32)
        self.layer2 = StochasticBinaryLayer(32, 16)
        self.layer3 = StochasticBinaryLayer(16, 3)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        x = self.layer2(x, with_grad)
        x = self.layer3(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        self.layer2.get_grad(loss)
        self.layer3.get_grad(loss)
    
    def predict(self,x):
        x = torch.from_numpy(x).type(torch.FloatTensor)
        #Apply softmax to output.
        pred = F.softmax(self.forward(x), dim=1)

        ans = []
        for prediction in pred:
            ans.append(prediction.argmax().item())
        return ans
 
def run_model(num_epochs=100, batch_size=1, learning_rate=0.001, train_loader = None, test_loader = None):
    model = IrisModel().cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Move tensors to the configured device
            inputs = batch['input'].float().cuda()
            labels = batch['label'].cuda()
            # Forward pass
            outputs = model(inputs)
            # One hot encoding buffer that you create out of the loop and just keep reusing
            #labels_onehot = torch.FloatTensor(batch_size, num_classes)

            # In your for loop
            #labels_onehot.zero_()
            #labels_onehot.scatter_(1, (labels.long()).view(-1,1), 1)

            #loss = torch.sum((outputs - labels_onehot.cuda())**2) / batch_size
            #print("Inputs: ", inputs)
            #print("Labels: ", labels)
            #print("Outputs: ", outputs)
            #print()
            loss = criterion(outputs, labels).cuda()
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
        inputs = batch['input'].float().cuda()
        labels = batch['label'].cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on iris data: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'models/iris_model.ckpt')
    print("Model saved to: ", os.getcwd() + "/models/iris_model.ckpt")

if __name__ == "__main__":
    from load_iris import getIrisDataLoader

    trainData, testData, train, test = getIrisDataLoader()