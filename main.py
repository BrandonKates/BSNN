import torch
import torch.nn as nn
import numpy as np

from dataloaders import linear_data, xor_data, mnist_data
from models import linear, bernoulli
from parser import Parser
from run_model import run_model

def get_data(args):
    if args.dataset == 'linear':
        train_data, test_data, train_loader, test_loader = linear_data.get(n=args.num_samples, d=args.input_size, sigma=0.15, test_split=0.2, batch_size=args.batch_size, num_workers=1)

    elif args.dataset in ['xor','XOR']:
        train_data, test_data, train_loader, test_loader = xor_data.get(n=args.num_samples, d=args.input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers=1)

    elif args.dataset in ['mnist', 'MNIST']:
        train_data, test_data, train_loader, test_loader = mnist_data.get(test_split = 0.2, batch_size = args.batch_size, num_workers=1)
    
    return train_data, test_data, train_loader, test_loader

def construct_model(args, output_size):
    if args.model == "linear":
        return linear.LinearModel(args.input_size, [1,2], output_size)

    elif args.model == "bernoulli":
        return bernoulli.BernoulliModel(args.input_size, [6 ,5, 6], output_size)

def main():
    args = Parser().parse()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(1)
    train_data, test_data, train_loader, test_loader = get_data(args)
    print(train_data)
    print("Train Data Shape: ", train_data.data.shape)
    print("Test Data Shape: ", train_data.targets.shape)
    # labels should be a whole number from [0, num_classes - 1]
    output_size = int(max(max(train_data.targets), max(test_data.targets))) + 1
    model =  construct_model(args, output_size).to(device)
    criterion = nn.CrossEntropyLoss() #TODO : make generic
    run_model(model, args, criterion, train_loader, test_loader, device)

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
