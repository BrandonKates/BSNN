import torch
import torch.nn.functional as F

from dataloaders import cifar10, mnist, svhn
from models import lenet5, simpleconv, complexconv, resnet, vgg
from parser import Parser
from main import get_data
from run_model import model_grads, model_temps
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import layers as L

def get_brier_score(outputs, labels, device):
    num_classes = outputs.shape[1]
    one_hot = torch.zeros(labels.size(0), num_classes).to(device).scatter_(1, labels.long().view(-1, 1).data, 1)
    mask = one_hot.gt(0)
    loss_label = torch.masked_select(outputs, mask)
    loss_label = (1 - loss_label)*(1 - loss_label)
    loss_other = torch.masked_select(outputs, ~mask)
    loss_other = loss_other*loss_other
    loss = (loss_label.sum() + loss_other.sum()) / outputs.shape[0] / outputs.shape[1]
    return loss.item()


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
    print('Plotting for num passes ' + str(num_passes))        
    model.eval()
    test_loss = 0
    brier_score = 0
    correct = 0
    bins_accuracy = [[] for _ in range(10)]
    bins_confidence = [[] for _ in range(10)]
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(num_passes):
                outputs.append(model(inputs))

            mean_output = torch.mean(torch.stack(outputs), dim=0)
            test_loss += criterion(mean_output, labels).sum().item()
            pred = mean_output.argmax(dim=1)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            softmaxed = torch.nn.Softmax(dim=1)(mean_output)
            brier_score += get_brier_score(softmaxed, labels, device)
            confidence = torch.max(softmaxed, dim=1)[0]
            for i in range(len(confidence)):
                bins_accuracy[min(int(confidence[i]*10), 9)].append((pred[i] == labels[i]).item())
                bins_confidence[min(int(confidence[i]*10), 9)].append((pred[i]).item())
                                

    calc_calibration_error(bins_confidence, bins_accuracy)
    plot_calibration_accuracy(bins_accuracy, args.model + "_" + str(num_passes))
    print(f'Correct: {correct} Test Loss: {test_loss}, Brier Score: {brier_score}\n')


def FGSM(model, test_loader, device, sample_num=1, epsilon=0.1):
    criterion = torch.nn.CrossEntropyLoss()
    adv_correct = 0
    correct = 0
    total = 0
    avg_entropy = 0 
    for i, (images,labels) in enumerate(test_loader):
        min_val = images.min().item()
        max_val = images.max().item()
        if torch.cuda.is_available():
            images = images.to(device)
            labels = labels.to(device)
        images.requires_grad = True
        # generate adversarial 
        outputs = model(images)
        loss = criterion(outputs,labels)
        model.zero_grad()
        if images.grad is not None:
            images.grad.data.fill_(0)
        loss.backward()
        grad = images.grad.data 
        for i in range(sample_num - 1):
            outputs = model(images)
            loss = criterion(outputs,labels)
            model.zero_grad()
            if images.grad is not None:
                images.grad.data.fill_(0)
            loss.backward()
            grad += images.grad.data 
        #print(f'FGSM grad: {torch.norm(images.grad)}')
        #print(f'model grads: {model_grads(model)}')
        grad = torch.sign(grad) # Take the sign of the gradient.
        images_adv = torch.clamp(images.data + epsilon*grad,min_val,max_val) # x_adv = x + epsilon*grad
        adv_output = F.softmax(model(images_adv), dim = 1).data  # output with adverserial noise
        for i in range(sample_num-1):
            adv_output += F.softmax(model(images_adv), dim = 1).data
        adv_output = adv_output / (sample_num)
        _,predicted = torch.max(outputs.data,1)
        _,adv_predicted = torch.max(adv_output.data,1) # Prediction on the image after adding adverserial noise
        total += labels.size(0)
        adv_correct += (adv_predicted == labels).sum().item()
        correct += (predicted == labels).sum().item()
    print(adv_correct/total)
    return adv_correct/total


def adv_robust(model, testloader, sample_num, device):
    for i in range(21):
        eps = i*0.005
        FGSM(model, testloader, device, sample_num=sample_num, epsilon=eps)


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
    print("Test Data Shape: ", test_data.data.shape)
    if args.deterministic:
        calc_calibration(args, model, device, test_loader, args.batch_size, num_labels, 1)
        adv_robust(model, test_loader, 1, device)
    else:
        for num_passes in [1, 5, 10]:
            calc_calibration(args, model, device, test_loader, args.batch_size, num_labels, num_passes)
            adv_robust(model, test_loader, num_passes, device)


if __name__ == '__main__':
    main()
