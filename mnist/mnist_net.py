import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNISTNet(torch.nn.Module):


    def __init__(self):
        super(MNISTNet, self).__init__()
        self.input_layer = torch.nn.Linear(28*28, 300)
        self.hidden_layer = torch.nn.Linear(300, 10)


    def forward(self, x):
        activation = self.hidden_layer(self.input_layer(x))
        return torch.nn.functional.softmax(activation, dim=0)


