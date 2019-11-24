import torch
import torch.nn as nn
import numpy as np

from dataloaders import linear_data, xor_data, mnist_data, spiral_data
from models import linear, bernoulli
from parser import Parser
from run_model import run_model
from math import log, ceil

def get_data(args):
    if args.dataset == 'linear':
        train_data, test_data, train_loader, test_loader = linear_data.get(n=args.num_samples, d=args.input_size, sigma=0.15, test_split=0.2, batch_size=args.batch_size, num_workers=1)

    elif args.dataset in ['xor','XOR']:
        train_data, test_data, train_loader, test_loader = xor_data.get(n=args.num_samples, d=args.input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers=1)

    elif args.dataset in ['mnist', 'MNIST']:
        train_data, test_data, train_loader, test_loader = mnist_data.get(test_split = 0.2, batch_size = args.batch_size, num_workers=1)

    elif args.dataset == 'spiral':
        train_data, test_data, train_loader, test_loader = spiral_data.get(n=args.num_samples, test_split=.2,batch_size=args.batch_size)
    
    return train_data, test_data, train_loader, test_loader

def construct_model(args, output_size):
    hidden_layers = [int(i) for i in args.hidden_layers]
    if args.model == "linear":
        return linear.LinearModel(args.input_size, hidden_layers, output_size)

    elif args.model == "bernoulli":
        return bernoulli.BernoulliModel(args.input_size, hidden_layers, output_size)

def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    train_data, test_data, train_loader, test_loader = get_data(args)

    print("Using device: ", device)
    print(train_data)
    print("Train Data Shape: ", train_data.data.shape)
    print("Test Data Shape: ", train_data.targets.shape)
    # labels should be a whole number from [0, num_classes - 1]
    num_labels = int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = int(ceil(log(num_labels, 2)))
    model =  construct_model(args, output_size).to(device)
    print("Model: ", model)
#    criterion = nn.MSELoss() #TODO : make generic
    def criterion(x,y):
        return torch.sum(1 - torch.eq(x,y))
    run_model(model, args, criterion, train_loader, test_loader, num_labels, device)

if __name__ == '__main__':
    main()
