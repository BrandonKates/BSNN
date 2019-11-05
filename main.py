import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np

from dataloaders import linear, xor
from models import linear_bernoulli, linear_binomial, xor_bernoulli, xor_model
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

        linear_bernoulli.run_model(train, test, 1, input_size, hidden_size,
                num_classes, num_epochs, batch_size, learning_rate, device)


    elif args.dataset == 'binomial':
        n = 100
        input_size = 2
        hidden_size = 1
        num_classes = 2
        num_epochs = 200
        batch_size = 1
        learning_rate = 0.001
        binomial_n = 1

        train_data, test_data, train, test = linear.get(n, num_classes, 0.15, 0.2, 1, 1)

        # result = linear_binomial.run_model(train, test, 1, input_size, hidden_size,
        #         num_classes, num_epochs, batch_size, learning_rate, device, binomial_n)

        # print('Accuracy of the network on linearly separable data: {} %'.format(result))


        all_results = []
        all_results.append(['num_epochs','binomial_n','result'])
        for num_epochs_i in range(1,1000):
            for binomial_n_j in range(1,50):
                result = linear_binomial.run_model(train, test, 1, input_size, hidden_size,
                    num_classes, num_epochs_i, batch_size, learning_rate, device, binomial_n_j)
                # 
                # print('Accuracy of the network on linearly separable data where #epoch {1} and n_binom {2}: {0} %'.format(result,num_epochs_i,binomial_n_j), )
                # 
                all_results.append([num_epochs_i,binomial_n_j, result])

  

        np.savetxt("results.csv", all_results, delimiter=",", fmt='%s')

    elif args.dataset == 'xor' or args.dataset == 'XOR':
        n=200
        input_size = 2
        hidden_size = 10
        num_classes = 1

        train_data, test_data, train_loader, test_loader = xor.get(n=n, d=input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers = 1)

        criterion = nn.MSELoss()
        xor_bernoulli.run_model(
            args, criterion, train_loader, test_loader, 
            device, input_size, hidden_size, num_classes)

    elif args.dataset == 'xor-model':
        n=200
        input_size = 2
        hidden_size = 10
        num_classes = 1

        train_data, test_data, train_loader, test_loader = xor.get(n=n, d=input_size, sigma = 0.25, test_split = 0.2, batch_size = args.batch_size, num_workers = 1)

        criterion = nn.CrossEntropyLoss()
        xor_model.run_model(
            args, criterion, train_loader, test_loader, 
            device, input_size, hidden_size, num_classes)       

if __name__ == '__main__':
    main()
