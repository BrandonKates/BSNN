import torch
import torch.nn as nn
import numpy as np

from dataloaders import linear, xor
from models import linear_bernoulli, linear_binomial, xor_bernoulli, xor_baseline
from parser import Parser

def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    train_data, test_data, train_loader, test_loader = get_data(args)
    print("Train Data Shape: ", train_data.inputs.shape)
    print("Test Data Shape: ", train_data.labels.shape)

def get_data(args):
    if args.dataset == 'linear':
        train_data, test_data, train_loader, test_loader = linear.get(n=args.num_samples, d=args.input_size, sigma=0.15, test_split=0.2, batch_size=args.batch_size, num_workers=1)

    elif args.dataset in ['xor','XOR']:
        train_data, test_data, train_loader, test_loader = xor.get(n=args.num_samples, d=args.input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers=1)
    
    return train_data, test_data, train_loader, test_loader

def construct_model(args,):
    if args.model == "linear":
        #criterion
        linear.run_model(args.criterion, train_loader, test_loader, device)


if __name__ == '__main__':
    main()

    '''
        criterion = nn.MSELoss()
        xor_bernoulli.run_model(
            args, criterion, train_loader, test_loader, 
            device, input_size, hidden_size, num_classes)


        criterion = nn.CrossEntropyLoss()
        xor_model.run_model(
            args, criterion, train_loader, test_loader, 
            device, input_size, hidden_size, num_classes)  
'''
