import torch
import torch.nn as nn
import numpy as np

from dataloaders import cifar10_data, mnist_data
from models import gumbel_conv_lecun, lenet5
from parser import Parser
from run_model import run_model


def get_data(args):
    if args.dataset in ['mnist', 'MNIST']:
        set_classes = [int(i) for i in args.set_classes] if args.set_classes else [0,1,2,3,4,5,6,7,8,9]
        return mnist_data.get(resize=args.resize_input, batch_size = args.batch_size)

    elif args.dataset == 'cifar10':
        return cifar10_data.get(args.batch_size, num_workers=0)

def construct_model(args, output_size, num_labels, device='cpu'):
    hidden_layers = [int(i) for i in args.hidden_layers]
    if args.model == 'gumbel-conv':
        return gumbel_conv_lecun.GumbelConvLecunModel(device=device,orthogonal=not args.no_orthogonal)

    elif args.model == 'lenet5':
        return lenet5.LeNet5(orthogonal = not args.no_orthogonal)

    
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
