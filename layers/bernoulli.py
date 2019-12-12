import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from copy import deepcopy
 
class BernoulliLayer(nn.Module):
    def __init__(self, input_dim, output_dim, new_loss_importance = 0.1, device="cpu"):
        super(BernoulliLayer, self).__init__()
        self.lin      = nn.Linear(input_dim,output_dim, bias=True)
        # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
        # We keep a running averave in order to compute the best loss correction to minmize estimator variance.
        self.cnum = torch.tensor(0.0).to(device)
        self.dnum = torch.tensor(0.25).to(device) #Assuming we're usually near 0.5
        self.new_loss_importance = new_loss_importance
        self.device = device
        self.weight_grads = []
        self.bias_grads = []
        self.squared_diff = []

    def forward(self, x, with_grad):
        
        l = self.lin(x)
        with torch.no_grad():
            p = torch.sigmoid(l)
        o = 2*torch.bernoulli(p)-1
        if with_grad:
            grad_cor = ((o+1)/2) - p
            #with torch.no_grad():
            self.squared_diff.append((grad_cor*grad_cor).mean())
            # See https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
            # This correctly takes care of exactly part of the gradient that does not depend on loss
            torch.sum(grad_cor*l).backward()

            self.weight_grads.append(deepcopy(self.lin.weight.grad.data))
            self.bias_grads.append(deepcopy(self.lin.bias.grad.data))

            self.zero_grad()

            '''
            print "X=",x, "W=",self.lin.weight.data.T, "b=",self.lin.bias.data
            print "eligibility=", grad_cor
            print "grad=", self.lin.weight.grad
            print "p=",p,"o=",o
            '''
        return o
    
    def get_grad(self, losses):
        #Should be a backward hook, I know, but come on. We will fix that a little later.
        # First, we compute the c to subtract,
        assert(len(losses) == len(self.weight_grads))
        assert(len(losses) == len(self.bias_grads))
        assert(len(losses) == len(self.squared_diff))

        for i in range(len(losses)):
            loss = losses[i].float()
            c = self.cnum / self.dnum
            self.cnum = (1-self.new_loss_importance)*self.cnum + self.new_loss_importance*loss*self.squared_diff[i]
            self.dnum = (1-self.new_loss_importance)*self.dnum + self.new_loss_importance*self.squared_diff[i]
        # Then, we subtract if from the loss
            correction = loss - c
        # And finally, we compute the gradients that stem from this loss.
            self.lin.weight.grad += correction*self.weight_grads[i]
            if type(self.lin.bias) != type(None):
               self.lin.bias.grad += correction*self.bias_grads[i]
        self.weight_grads = []
        self.bias_grads = []
        self.squared_diff = []
    
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
