import torch
import argparse

from run_model import test
from dataloaders import stl10
from models import vgg

def parse_args():
    parser = argparse.ArgumentParser('test cifar10 generalization on STL10')
    parser.add_argument('--inference-passes', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--saved-model', type=str, required=True)
    parser.add_argument('--deterministic', action='store_true', default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    trainset, testset, trainloader, testloader = stl10(args.batch_size)

    model = vgg.vgg16(not args.deterministic, device, False).to(device)
    saved_state = torch.load(args.saved_model)
    if args.saved_model[-4:] == '.tar':
        saved_state = saved_state['model_state_dict']
    model.load_state_dict(saved_state)

    criterion = torch.nn.CrossEntropyLoss()

    test_loss, pct_right = test(args, model, device, testloader, criterion, 10)
    print(f'test loss: {test_loss}, correct: {100*pct_right}')

if __name__ == '__main__':
    main()
