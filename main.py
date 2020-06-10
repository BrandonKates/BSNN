import sys 

import torch
import torch.nn as nn
import numpy as np

from dataloaders import mnist_data, cifar10_data
from models import lenet5, simpleconv, vgg, complexconv
from parser import Parser
from run_model import run_model


def get_data(args):
    if args.dataset == 'mnist':
        resize = args.resize_input
        batch_size = args.batch_size
        return mnist_data.get(resize=resize, batch_size=batch_size)

    elif args.dataset == 'cifar10':
        return cifar10_data.get(args.batch_size, num_workers=0)


def construct_model(args, device='cpu'):
    orthogonal = not args.no_orthogonal
    stochastic = not args.deterministic

    if args.model == 'lenet5':
        return lenet5.LeNet5(
                device=device, orthogonal=orthogonal, stochastic=stochastic)

    elif args.model == 'simpleconv':
        return simpleconv.SimpleConv(
                device=device,orthogonal=orthogonal,stochastic=stochastic)

    elif args.model == 'vgg':
        return vgg.vgg11(stochastic, device)

    elif args.model == 'complexconv':
        return complexconv.ComplexConv(
            device=device, 
            orthogonal=orthogonal,
            stochastic=stochastic)

    else:
        print(f"{args.model} is not a valid model")
        sys.exit(-1)

    
def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    train_data, test_data, train_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels
    model = construct_model(args, device).to(device)
    print("Model Architecture: ", model)
    print("Using device: ", device)
    print("Train Data Shape: ", train_data.data.shape)
    criterion = nn.CrossEntropyLoss()
    run_model(model, args, criterion, train_loader, test_loader, num_labels, device)

if __name__ == '__main__':
    main()
