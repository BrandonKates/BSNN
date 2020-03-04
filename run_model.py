import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from helpers import plot_decision_boundary

from sklearn.metrics import confusion_matrix

p_values = []

def _log_model_stats(model, debug=True):
    print('Final Layer Weights')
    torch.set_printoptions(profile="full")
    print(model.linear_layer.weight)
    print('Final Layer Bias')
    print(model.linear_layer.bias)
    torch.set_printoptions(profile="default")
    print('P values')
    print(model.layers[-1].p_avg)
    print('last layer activations')
    print(torch.mv(model.linear_layer.weight.double() , model.layers[-1].p_avg))
    if debug:
        p_values.append(model.layers[-1].p_avg.data)

def train(args, model, device, train_loader, optimizer, epoch, criterion, batch_size, num_passes):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.to(device)
        inputs = inputs.flatten(start_dim=1)
        labels = labels.long()
        optimizer.zero_grad()
        losses = [criterion(model(inputs, with_grad = True), labels) for _ in range(num_passes)]
        model.get_grad(losses)
        optimizer.step() 
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(np.array(list(map(lambda t: t.data, losses))))))
            
            _log_model_stats(model)
    print (p_values)

def test(args, model, device, test_loader, criterion, batch_size, num_labels, num_passes):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    correct = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.float().to(device), labels.to(device)
        inputs = inputs.flatten(start_dim=1)
        labels = labels.long()
        passes_pred = []
        for _ in range(num_passes):
            output = model(inputs, with_grad = False)
            test_loss += criterion(output, labels).sum().item() # sum up batch loss
            passes_pred.append(output.argmax(dim=1, keepdim=True))
        pred = torch.mode(torch.cat(passes_pred, dim=1), dim=1, keepdim=True)[0]
        correct += pred.eq(labels.view_as(pred)).sum().item() #torch.all(output.eq(labels)).sum().item()
        conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        )
    )
    print("Confusion Matrix:\n", np.int_(conf_mat))


def run_model(model, args, criterion, train_loader, test_loader, num_labels, device, t_passes, i_passes):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion, args.batch_size, t_passes)
        test(args, model, device, test_loader, criterion, args.batch_size, num_labels, 1)
        if epoch % 50 == 0:
            if (args.save_model):
                torch.save(model.state_dict(), args.save_location + str(epoch))

    if args.plot_boundary:
        plot_decision_boundary(model.predict(device, i_passes), test_loader.dataset.data, test_loader.dataset.targets, save_name=str(args.model) + str(args.dataset))
