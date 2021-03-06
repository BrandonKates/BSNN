import torch

from dataloaders import cifar10, mnist
from models import lenet5, simpleconv, complexconv
from parser import Parser
from run_model import run_model


def get_data(args):
    if args.dataset == 'mnist':
        resize = args.resize_input
        batch_size = args.batch_size
        return mnist(resize=resize, batch_size=batch_size)

    elif args.dataset == 'cifar10':
        return cifar10(args.batch_size, num_workers=0)


def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_data, test_data, train_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels

    init_args = [args.normalize, not args.deterministic, device]
    models = {
        'lenet5': lenet5.LeNet5,
        'simpleconv': simpleconv.SimpleConv,
        'complexconv': complexconv.ComplexConv
    }
    model = models[args.model](*init_args).to(device)

    print("Model Architecture: ", model)
    print("Using device: ", device)
    print("Train Data Shape: ", train_data.data.shape)
    print("Normalize layer outputs?: ", args.normalize)
    criterion = torch.nn.CrossEntropyLoss()
    run_model(model, args, criterion, train_loader, test_loader, num_labels, device)

if __name__ == '__main__':
    main()
