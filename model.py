import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

class StochasticBinaryLayer(nn.Module):
    def __init__(self, input_dim, output_dim, new_loss_importance = 0.1):
        super(StochasticBinaryLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        # We keep a running averave in order to compute the best loss correction to minmize estimator variance.
        self.cnum = torch.tensor(0.0)
        self.dnum = torch.tensor(0.25) #Assuming we're usually near 0.5
        self.last_squared_dif = torch.tensor(0)
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


def demonstrate():
    sl = StochasticBinaryLayer(2, 3).cuda()
    optimizer = torch.optim.Adam(sl.parameters(), lr=0.05)
    target = torch.tensor([0., 0., 1.]).cuda()
    for i in range(100):
        inp = torch.randn(50, 2).cuda()
        print("Starting New Minibatch")
        for i in range(20):
            optimizer.zero_grad()
            out = sl(inp)
            loss = torch.sum((out - target)**2)
            print(loss)
            sl.get_grad(loss)
            optimizer.step()
    

if __name__ == "__main__":
#    sl = StochasticBinaryLayer(2, 3).cuda()
#    print(sl.parameters())
#    print(sl.lin.parameters())
    demonstrate()

