import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

class StochasticBinaryLayer(nn.Module):
    def __init__(self, input_dim, output_dim, new_loss_importance = 0.1):
        super(StochasticBinaryLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        # We keep a running averave in order to compute the best loss correction to minmize estimator variance.
        self.cnum = torch.tensor(0.0).cuda()
        self.dnum = torch.tensor(0.25).cuda() #Assuming we're usually near 0.5
        self.last_squared_dif = torch.tensor(0).cuda()
        self.new_loss_importance = new_loss_importance
    

    def forward(self, x, with_grad=True):
        l = self.lin(x)
        with torch.no_grad():
            p = torch.sigmoid(l)
        o = torch.bernoulli(p)
        if with_grad:
            grad_cor = o - p
            with torch.no_grad():
                self.last_squared_dif = (grad_cor*grad_cor).mean()
            # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
            # This correctly takes care of exactly part of the gradient that does not depend on loss
            torch.sum(l*grad_cor).backward()
        return o
    
    def get_grad(self, loss):
        #Should be a backward hook, I know, but come on. We will fix that a little later.
        # First, we compute the c to subtract,
        c = self.cnum / self.dnum
        self.cnum = 0.9*self.cnum + 0.1*loss*self.last_squared_dif
        self.dnum = 0.9*self.dnum + 0.1*self.last_squared_dif
        # Then, we subtract if from the loss
        correction = loss - c
        # And finally, we compute the gradients that stem from this loss.
        self.lin.weight.grad *= correction
        if type(self.lin.bias) != type(None):
            self.lin.bias.grad *= correction
    
    def parameters(self):
        # Everythin else is not trainable
        return self.lin.parameters()
    
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

def run_model(input_size = 2, hidden_size=3, num_classes=2, num_epochs=5, batch_size=1, learning_rate=0.001, n=100, train_loader = None, test_loader = None):
    model = StochasticBinaryLayer(input_size, num_classes).cuda()
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Move tensors to the configured device
            inputs = batch['input'].cuda()
            labels = batch['label'].cuda()
            print("Labels: ", labels)
            # Forward pass
            outputs = model(inputs)
            print("Outputs: ", outputs)

            loss = criterion(outputs, labels)
            #loss = torch.sum((outputs - labels)**2)

            # Backward and optimize
            
            model.get_grad(loss)
            optimizer.step()

            # ignore loss, using only one example loss will either always be 0 or 1
            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    correct = 0
    total = 0
    for batch in test_loader:
        inputs = batch['input'].cuda()
        labels = batch['label'].cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print("Labels: ", labels)
        print("Outputs: ", outputs)
        print("Predicted: ", predicted)
        print("\n")

    print('Accuracy of the network on linearly separable data: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    print(model.state_dict())
    torch.save(model.state_dict(), 'model.ckpt')
    

if __name__ == "__main__":
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
    
    run_model(input_size = input_size,hidden_size=hidden_size, num_classes=num_classes, num_epochs=num_epochs,
        batch_size=batch_size, learning_rate=learning_rate, n=n,
        train_loader=train,
        test_loader=test)

