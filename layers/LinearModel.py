import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from load_linearData import getLinearDataLoader
import torch.nn.functional as F

from model import StochasticBinaryLayer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 2
hidden_size = 10
num_classes = 2
num_epochs = 5
batch_size = 1
learning_rate = 0.001
n=100

train_dataset, test_dataset, train_loader, test_loader = \
    getLinearDataLoader(n=n, d=input_size, sigma = 0.15, test_split = 0.2, batch_size = 1, num_workers = 4)


# Fully connected neural network with one hidden layer
class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        #self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        #out = self.fc2(out)
        return out
#This function takes an input and predicts the class, (0 or 1)        
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

def run_model(input_size = 2, hidden_size=3, num_classes=2, num_epochs=5, batch_size=1, learning_rate=0.001, n=100, train_loader = train_loader, test_loader = test_loader):
    #model = LinearNet(input_size, hidden_size, num_classes).to(device)
    model = StochasticBinaryLayer(input_size, num_classes)#.cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Move tensors to the configured device
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on linearly separable data: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    print(model.state_dict())
    torch.save(model.state_dict(), 'model.ckpt')
    
if __name__ == '__main__':
    run_model()
