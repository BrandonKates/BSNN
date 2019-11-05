import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

class BinomialLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n=1, new_loss_importance = 0.1, device="cpu"):
        super(BinomialLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        # We keep a running averave in order to compute the best loss correction to minmize estimator variance.
        self.cnum = torch.tensor(0.0).to(device)
        self.dnum = torch.tensor(0.25).to(device) #Assuming we're usually near 0.5
        self.last_squared_dif = torch.tensor(0).float().to(device)
        self.new_loss_importance = new_loss_importance.to(device)
        self.device = device
        self.n = n

    def forward(self, x, with_grad=True):
        result = self.lin(x)
        #with torch.no_grad():
        p = torch.sigmoid(result) # output of sigmoid is [0,1], so we use this function as squashing function to get a prob for bernoulli
        
        k = torch.distributions.binomial.Binomial(total_count=self.n, probs=p).sample().to(self.device)

        if with_grad:
            grad_cor = k - self.n*p
            with torch.no_grad():
                self.last_squared_dif += (grad_cor*grad_cor).mean()
            # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
            # This correctly takes care of exactly part of the gradient that does not depend on loss
            torch.sum(result*grad_cor).backward()
        return k
    
    def get_grad(self, loss):
        #Should be a backward hook, I know, but come on. We will fix that a little later.
        # First, we compute the c to subtract,
        loss = loss.float()
        c = self.cnum / self.dnum
        self.cnum = 0.9*self.cnum + 0.1*loss*self.last_squared_dif
        self.dnum = 0.9*self.dnum + 0.1*self.last_squared_dif
        self.last_squared_dif = torch.tensor(0).float().to(self.device)

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
