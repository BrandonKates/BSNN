import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from dataloaders import linear, xor
from models import linear_bernoulli, linear_binomial, xor_bernoulli
from parser import Parser

def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)

    if args.dataset == 'linear':
        n = 100
        input_size = 2
        hidden_size = 1
        num_classes = 2
        num_epochs = 200
        batch_size = 1
        learning_rate = 0.001


        train_data, test_data, train, test = linear.get(n, num_classes, 0.15, 0.2, 1, 1)

        linear_bernoulli.run_model(train, test, input_size, hidden_size,
                num_classes, num_epochs, batch_size, learning_rate, device)


    elif args.dataset == 'binomial':
        n = 100
        input_size = 2
        hidden_size = 1
        num_classes = 2
        num_epochs = 200
        batch_size = 1
        learning_rate = 0.001

        train_data, test_data, train, test = linear.get(n, num_classes, 0.15, 0.2, 1, 1)

        linear_binomial.run_model(train, test, input_size, hidden_size,
                num_classes, num_epochs, batch_size, learning_rate, device)

    elif args.dataset == 'xor' or args.dataset == 'XOR':
        n=100
        input_size = 2
        hidden_size = 1
        num_classes = 2
        learning_rate = 0.001

        train_data, test_data, train_loader, test_loader = xor.get(n=n, d=input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers = 1)

        criterion = nn.MSELoss()
        xor_bernoulli.run_model(
            args, criterion, train_loader, test_loader, 
            device, input_size, hidden_size, num_classes)

if __name__ == '__main__':
    main()
