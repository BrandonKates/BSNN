import torch

from dataloaders import cifar10, mnist, svhn
from models import lenet5, simpleconv, complexconv, resnet, vgg
from parser import Parser
from main import get_data
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def plot_calibration_accuracy(bins, id):
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
    plt.clf()

def calc_calibration_error(bins_confidence, bins_accuracy):
    num_samples = list(map(lambda x : len(x), bins_accuracy))
    bins_avg_accuracy = list(map(lambda x: 0 if len(x) == 0 else np.mean(x), bins_accuracy))
    bins_avg_confidence = list(map(lambda x: 0 if len(x) == 0 else np.mean(x), bins_confidence))
    ece = sum(np.array(num_samples)*np.abs(np.array(bins_avg_accuracy) - np.array(bins_avg_confidence)))/sum(num_samples)
    mce = max(np.abs(np.array(bins_avg_accuracy) - np.array(bins_avg_confidence)))
    print("ECE: {:.3f}, MCE: {:.3f}".format(ece, mce))

def calc_calibration(args, model, device, test_loader, batch_size, num_labels, num_passes):
    print('Plotting for num passes' + str(num_passes))        
    model.eval()
    bins_accuracy = [[] for _ in range(10)]
    bins_confidence = [[] for _ in range(10)]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(num_passes):
                outputs.append(model(inputs))

            mean_output = torch.mean(torch.stack(outputs), dim=0)
            pred = mean_output.argmax(dim=1)
            confidence = torch.max(torch.nn.Softmax(dim=1)(mean_output), dim=1)[0]
            for i in range(len(confidence)):
                bins_accuracy[min(int(confidence[i] * 10), 9)].append((pred[i] == labels[i]).item())
                bins_confidence[min(int(confidence[i] * 10), 9)].append((pred[i]).item())
                                

    calc_calibration_error(bins_confidence, bins_accuracy)
    plot_calibration_accuracy(bins_accuracy, args.model + "_" + str(num_passes))

def calc_calibration_det(args, model, device, test_loader, batch_size, num_labels):
    model.eval()
    bins_accuracy = [[] for _ in range(10)]
    bins_confidence = [[] for _ in range(10)]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            mean_output = model(inputs)
            pred = mean_output.argmax(dim=1)
            confidence = torch.max(torch.nn.Softmax(dim=1)(mean_output), dim=1)[0]
            for i in range(len(confidence)):
                bins_accuracy[min(int(confidence[i] * 10), 9)].append((pred[i] == labels[i]).item())
                bins_confidence[min(int(confidence[i] * 10), 9)].append((pred[i]).item())
                                

    calc_calibration_error(bins_confidence, bins_accuracy)
    plot_calibration_accuracy(bins_accuracy, args.model + "_det")



def main():
    args = Parser().parse()

    use_cuda = not args.cpu and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    data = get_data(args)
    if len(data) == 4:
        train_data, test_data, train_loader, test_loader = data
    elif len(data) == 5:
        train_data, test_data, train_loader, val_loader, test_loader = data

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = 10 #int(max(max(train_data.targets), max(test_data.targets))) + 1
    output_size = num_labels


    if 'resnet' in args.model:
        constructor = getattr(resnet, args.model)
        model = constructor(not args.deterministic, device).to(device)

    elif 'vgg' in args.model:
        constructor = getattr(vgg, args.model)
        model = constructor(not args.deterministic, device, args.orthogonal).to(device)

    else:
        init_args = [args.normalize, not args.deterministic, device]
        models = {
            'lenet5': lenet5.LeNet5,
            'simpleconv': simpleconv.SimpleConv,
            'complexconv': complexconv.ComplexConv
        }
        model = models[args.model](*init_args).to(device)

    saved_state = torch.load(args.resume)
    if args.resume[-4:] == '.tar':
        saved_state = saved_state['model_state_dict']

    model.load_state_dict(saved_state)
    print("Model Architecture: ", model)
    print("Using device: ", device)
    print("Train Data Shape: ", train_data.data.shape)
    print("Normalize layer outputs?: ", args.normalize)
    if args.deterministic:
        calc_calibration_det(args, model, device, test_loader, args.batch_size, num_labels)
    else:
        for num_passes in [1, 5, 10]:
            calc_calibration(args, model, device, test_loader, args.batch_size, num_labels, num_passes)


if __name__ == '__main__':
    main()
