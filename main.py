import torch
import torch.nn as nn
import numpy as np

from dataloaders import cifar10_data, mnist_data
from models import gumbel_conv_lecun, lenet5, densenet_k12_L40, gumbel_conv_vgg, simple_conv
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
    if args.model == 'gumbel-conv':
        return gumbel_conv_lecun.GumbelConvLecunModel(
                device=device,
                orthogonal=orthogonal)

    if args.model == 'simple-conv':
        return simple_conv.SimpleConv(device=device,
                orthogonal=orthogonal,stochastic=True)

    if args.model == 'vgg':
        return gumbel_conv_vgg.GumbelConvVGGModel(
                device=device,
                orthogonal=orthogonal)

    elif args.model == 'lenet5':
        return lenet5.LeNet5(orthogonal=orthogonal)

    elif args.model == 'densenet':
        return densenet_k12_L40.DenseNet(device=device)

    
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
