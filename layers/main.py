import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"



from model import StochasticBinaryLayer
from load_it import x_train, x_test, y_train, y_test

x_train = x_train.cuda()
x_test = x_test.cuda()
y_train = y_train.cuda()
y_test = y_test.cuda()


print(x_train)

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = StochasticBinaryLayer(input_dim,50)
        self.layer2 = StochasticBinaryLayer(50, 50)
        self.layer3 = StochasticBinaryLayer(50, 3)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        x = self.layer2(x, with_grad)
        x = self.layer3(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer3.get_grad(loss)
        self.layer1.get_grad(loss)
        self.layer2.get_grad(loss)

if __name__ == "__main__":
    m = Model(4).cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.05)
    for i in range(100000):
        optimizer.zero_grad()
        pred = m(x_train)
        loss = torch.sum((y_train - pred)**2)/y_train.size(0)
        print("Train: ", loss)
        m.get_grad(loss)
        optimizer.step()
        pred2 = m(x_test, False)
        loss2 = torch.sum((y_test - pred2)**2)/y_test.size(0)
        print("Test: ", loss2)
        optimizer.zero_grad()
       
 

