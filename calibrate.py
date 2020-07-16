import torch

from dataloaders import cifar10, mnist, svhn
from models import lenet5, simpleconv, complexconv, resnet, vgg
from parser import Parser
from main import get_data
from matplotlib import pyplot as plt

def plot_calibration(bins, id):
    num_samples = list(map(lambda x : len(x), bins))
    accuracy = list(map(lambda x : 0 if len(x) == 0 else(1.0*sum(x))/len(x), bins))
    bin_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.plot(bin_vals, accuracy)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_calibration_'+id+'.png')
    plt.clf()
    plt.plot(bin_vals, num_samples)
    plt.xlabel('Confidence')
    plt.ylabel('Num Samples')
    plt.savefig('distribution_calibration_'+id+'.png')

def calc_calibration(args, model, device, test_loader, batch_size, num_labels, num_passes):
    print('Plotting for num passes' + str(num_passes))        
    model.eval()
    correct = 0
    bins = [[] for _ in range(10)]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            passes_pred = []
            passes_probab = []
            for _ in range(num_passes):
                output = model(inputs)
                passes_pred.append(output.argmax(dim=1, keepdim=True))
                passes_probab.append(torch.max(torch.nn.Softmax(dim=1)(output), dim=1)[0])

            pred = torch.mode(torch.cat(passes_pred, dim=1), dim=1, keepdim=False)[0]
            confidence = torch.mean(torch.stack(passes_probab), dim=0, keepdim=True)[0]

            for i in range(len(confidence)):
                bins[int(confidence[i] * 10)].append((pred[i] == labels[i]).item())
            break
            
    plot_calibration(bins, str(num_passes))



def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    train_data, test_data, train_loader, test_loader = get_data(args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = 10 #int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels


    if 'resnet' in args.model:
        constructor = getattr(resnet, args.model)
        model = constructor(not args.deterministic, device).to(device)

    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model = constructor(not args.deterministic, device).to(device)

    else:
        init_args = [args.normalize, not args.deterministic, device]
        models = {
            'lenet5': lenet5.LeNet5,
            'simpleconv': simpleconv.SimpleConv,
            'complexconv': complexconv.ComplexConv
        }
        model = models[args.model](*init_args).to(device)

    model.load_state_dict(torch.load(args.load_location))
    print("Model Architecture: ", model)
    print("Using device: ", device)
    print("Train Data Shape: ", train_data.data.shape)
    print("Normalize layer outputs?: ", args.normalize)
    for num_passes in [1, 5, 10]:
        calc_calibration(args, model, device, test_loader, args.batch_size, num_labels, num_passes)


if __name__ == '__main__':
    main()
