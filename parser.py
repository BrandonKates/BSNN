import argparse

from optim import *

class Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train BSNN Model')
        self.parser.add_argument('experiment_name')
        self.parser.add_argument('--dataset', '-d', type=str, required=True, 
                            help='which dataset do you want to train on?')

        self.parser.add_argument('--model', '-m', type=str, required=True, 
                            help='which model do you want to run?')

        self.parser.add_argument('--epochs', type=int, default=100,
                            help='number of epochs to train (default: 100)')

        self.parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate (default: 0.01)')

        self.parser.add_argument('--momentum', type=float, default=0.5,
                            help='SGD momentum')

        self.parser.add_argument('--cpu', action='store_true', default=False,
                            help='disables CUDA training')
        self.parser.add_argument('--gpu', type=int, default=0,
                help='index of GPU to use')

        self.parser.add_argument('--resize-input', action='store_true', 
                                default=False)

        self.parser.add_argument('--seed', type=int, default=1,
                            help='random seed')

        self.parser.add_argument('--log-interval', type=int, default=10,
                            help='how many batches to wait before logging training status')

        self.parser.add_argument('--save-model', '-s', action='store_true', default=False,
                            help='For Saving the current Model')

        self.parser.add_argument('--resume', '-r', type=str, 
                                    help='Location to Load Model')
        self.parser.add_argument('--no-save', action='store_true',
                    default=False, help='pass to prevent model saving/checkpoint')

        self.parser.add_argument('--deterministic', action='store_true', 
                default=False, 
                help='Run deterministic variant, if one exists')

        self.parser.add_argument('--print-grads', '-g',
        action='store_true', default=False, 
        help='print layer gradients if model implements `print_grads` ')

        self.parser.add_argument('--batch-size', type=int, default=64,
            help='input batch size for training')

        self.parser.add_argument('--inference-passes', '-i', type=int, default=10,
                help='number of forward passes during test')

        self.parser.add_argument('--normalize', '-n', default=False,
            action='store_true', help='batch norm, if model allows')
        self.parser.add_argument('--orthogonal', action='store_true', default=False)
        self.parser.add_argument('--adjust-lr', action='store_true',
                default=False)

        # temperature schedule arguments
        self.parser.add_argument('--temp-jang', '-tj',
                action='store_true')
        self.parser.add_argument('--temp-step', type=int, default=1000)
        self.parser.add_argument('--temp-exp', type=float, default=1e-2)
        self.parser.add_argument('--temp-limit', type=float, default=.5)
        self.parser.add_argument('--temp-const', type=float, default=1.)

        # densenet related arguments
        self.parser.add_argument('--layers', type=int, default=100)
        self.parser.add_argument('--growth', type=int, default=12)
        self.parser.add_argument('--reduction', type=float, default=.5)
        self.parser.add_argument('--no-bottleneck',
                action='store_true', dest='bottleneck')
        self.parser.set_defaults(bottleneck=True)
        
        self.parser.add_argument('--metrics-dir', type=str, default='runs')
        self.parser.add_argument('--log-dir', type=str, default='log')

    def parse(self):
        return self.parser.parse_args()

