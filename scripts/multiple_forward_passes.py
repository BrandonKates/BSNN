from argparse import ArgumentTypeError, ArgumentParser
from os import path

import torch
import torch.nn as nn

import dataloaders
from models import resnet
from run_model import test

def get_args():
    def validator(module):
        def _(arg):
            if not hasattr(module, arg):
                raise ArgumentTypeError(f'{arg} not in {module}')
            return getattr(module, arg)
        return _
    model = validator(resnet)
    dataset = validator(dataloaders)
    _path = lambda x: x if path.exists(x) else ArgumentTypeError(f'{x} not found')

    parser = ArgumentParser('run mfp experiments with saved models')

    parser.add_argument('model', help='model class trained', type=model)
    parser.add_argument('model_path', help='path to saved model', type=_path)
    parser.add_argument('dataset', help='dataset used', type=dataset)
    parser.add_argument('inference_passes', type=int)

    parser.add_argument('--gpu', type=int, help='gpu index', default=0)
    parser.add_argument('--seed', '-s', type=int, help='random seed', default=1)
    parser.add_argument('--batch-size', '-b', type=int, default=64)
    parser.add_argument('--num-workers', '-n', type=int, default=4)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}')
    torch.manual_seed(args.seed)

    testloader = (args.dataset(args.batch_size, args.num_workers))[3]
    criterion = torch.nn.CrossEntropyLoss()

    model = args.model(True, device).to(device)
    model.load_state_dict(torch.load(args.model_path))

    test(args, model, device, testloader, criterion, args.batch_size, 10)

if __name__ == '__main__':
    main()
