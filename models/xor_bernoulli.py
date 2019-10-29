import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layers import bernoulli

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device='cpu'):
        super(Net, self).__init__()
        self.layer1 = bernoulli.BernoulliLayer(input_size, hidden_size, device=device)
        self.layer2 = bernoulli.BernoulliLayer(hidden_size, num_classes, device=device)
        
    def forward(self, x, with_grad=True):
        x = self.layer1(x, with_grad)
        x = self.layer2(x, with_grad)
        return x

    def get_grad(self, loss):
        self.layer1.get_grad(loss)
        self.layer2.get_grad(loss)
    
    def predict(self,x):
        x = torch.from_numpy(x).type(torch.FloatTensor).cuda()
        #Apply softmax to output.
        pred = F.softmax(self.forward(x), dim=1)

        ans = []
        for prediction in pred:
            ans.append(prediction.argmax().item())
        return ans
