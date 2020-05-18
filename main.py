import torch
import torch.nn as nn
import numpy as np

from dataloaders import circle_data, linear_data, xor_data, spiral_data, cifar10_data, mnist_data
from models import linear, bernoulli, gumbel, gumbel_conv_lecun
from parser import Parser
from run_model import run_model
import sys
from math import log, ceil


def get_data(args):
    if args.dataset == 'linear':
        train_data, test_data, train_loader, test_loader = linear_data.get(n=args.num_samples, d=args.input_size, sigma=0.15, test_split=0.2, batch_size=args.batch_size, num_workers=1)
        
    if args.dataset == 'circle':
        train_data, test_data, train_loader, test_loader = circle_data.get(n=args.num_samples, d=args.input_size, num_labels=args.num_labels, test_split=0.2, batch_size=args.batch_size, num_workers=1)
        
    elif args.dataset in ['xor','XOR']:
        train_data, test_data, train_loader, test_loader = xor_data.get(n=args.num_samples, d=args.input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers=1)

    elif args.dataset in ['mnist', 'MNIST']:
        set_classes = [int(i) for i in args.set_classes] if args.set_classes else [0,1,2,3,4,5,6,7,8,9]
        train_data, test_data, train_loader, test_loader = mnist_data.get(resize=True, test_split = 0.2, batch_size = args.batch_size, num_workers=1, classes=set_classes)

    elif args.dataset == 'spiral':
        train_data, test_data, train_loader, test_loader = spiral_data.get(n=args.num_samples, test_split=.2,batch_size=args.batch_size)

    elif args.dataset == 'cifar10':
        train_data, test_data, train_loader, test_loader = cifar10_data.get(args.batch_size, num_workers=0)

    return train_data, test_data, train_loader, test_loader

def construct_model(args, output_size, num_labels, device='cpu'):
    hidden_layers = [int(i) for i in args.hidden_layers]
    if args.model == "linear":
        return linear.LinearModel(args.input_size, hidden_layers, output_size)

    elif args.model == "bernoulli":
        return bernoulli.BernoulliModel(args.input_size, hidden_layers, output_size, num_labels, device=device, orthogonal=not args.no_orthogonal)

    elif args.model == "gumbel":
        if args.temp == None:
            print('Must provide --temp parameter')
            sys.exit(-1)
        return gumbel.GumbelModel(args.input_size, hidden_layers, output_size, num_labels, args.temp, device=device, orthogonal=not args.no_orthogonal)

    elif args.model == 'gumbel-conv':
        return gumbel_conv_lecun.GumbelConvLecunModel(device=device,orthogonal=not args.no_orthogonal)

    
def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    train_data, test_data, train_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels
    model = construct_model(args, output_size, num_labels, device).to(device)
    print("Model Architecture: ", model)
    print("Using device: ", device)
    print("Train Data Shape: ", train_data.data.shape)
    #print("Test Data Shape: ", train_data.targets.shape)
    criterion = nn.CrossEntropyLoss()
    run_model(model, args, criterion, train_loader, test_loader, num_labels, device, args.t_passes, args.i_passes)

if __name__ == '__main__':
    main()
