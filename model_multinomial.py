import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

# change this if you have CUDA enabled
device = torch.device("cpu")

class StochasticBinaryLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_samples=1, new_loss_importance = 0.1):
        super(StochasticBinaryLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        # We keep a running averave in order to compute the best loss correction to minmize estimator variance.
        self.cnum = torch.tensor(0.0, device=device).cuda()
        self.num_samples = num_samples
        self.dnum = torch.tensor(0.25, device=device).cuda() #Assuming we're usually near 0.5
        self.last_squared_dif = torch.tensor(0, device=device)
        self.new_loss_importance = new_loss_importance
    

    def forward(self, x, with_grad=True):
        l = self.lin(x)
        with torch.no_grad():
            p = torch.sigmoid(l)
        s = [self.num_samples] + list(p.size())
        # Now, expand to multiple samples
        o = torch.bernoulli(p.expand(s))
        if with_grad:
            grad_cor = o - p
            with torch.no_grad():
                self.last_squared_dif = (grad_cor*grad_cor).mean()
            # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
            # This correctly takes care of exactly part of the gradient that does not depend on loss
            torch.sum(l*grad_cor).backward()
        return o.sum(0)/self.num_samples
    
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

def demonstrate(target):
    sl = StochasticBinaryLayer(2, 1)#.cuda()
    optimizer = torch.optim.Adam(sl.parameters(), lr=0.05)
    target = torch.tensor([0., 0., 1.], device=device)#.cuda()
    for i in range(100):
        inp = torch.randn(50, 2, device=device)#.cuda()
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

