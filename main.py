from dataloaders import linear, xor
from models import linear_bernoulli 

import torch

import argparse

parser = argparse.ArgumentParser(description='train a stochastic model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', '-d', type=str, required=True, 
                    help='which dataset do you want to train on?')

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



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
