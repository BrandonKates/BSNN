import logging
import math
from os import path, mkdir

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from optim import JangScheduler, ConstScheduler
import layers as L

def adjust_lr(base_lr, epoch, optimizer):
    lr = base_lr * (0.1 ** (epoch // 150)) * (.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def model_grads(model):
    grads = []
    for m in model.modules():
        if isinstance(m, L.Conv2d) or isinstance(m, L.Linear):
            grads.append(torch.norm(m.inner.weight.grad).item())
    return grads


def model_temps(model, val_only=True):
    temps = []
    for m in model.modules():
        if isinstance(m, L.Conv2d) or isinstance(m, L.Linear):
            if val_only:
                temps.append(m.temp.val)
            else:
                temps.append(m.temp)
    return temps


def checkpoint(model, optimizer, epoch, exp_name):
    to_pickle = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(to_pickle, f'checkpoints/{exp_name}_{epoch}.tar')


def avg(l):
    if len(l) == 0:
        return 0
    return sum(l)/len(l)

def record_metrics(writer, epoch, phase, **metrics):
    for metric_name, metric_val in metrics.items():
        writer.add_scalar(f'{phase}/{metric_name}', metric_val, epoch)
    
def log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss, temp):
    fmt = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTemp: {:.6f}'
    grads = model_grads(model)
    mean_grad = sum(grads)/len(grads)
    grads = '\tGrads: {:.6f}'.format(mean_grad)
    log_str = fmt.format(epoch, inputs_seen, inputs_tot, pct, loss, temp)
    logging.info(log_str + grads)

def log_test(avg_loss, correct, num_test_samples, conf_mat):
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
    pct_correct = 100. * correct / num_test_samples
    log_str = fmt.format(avg_loss, correct, num_test_samples, pct_correct)
    logging.info(log_str)
    logging.info(f'Confusion Matric:\n{np.int_(conf_mat)}')


def train(args, model, device, train_loader, optimizer, epoch, criterion,
        metrics_writer=None, temp_schedule=None):
    model.train()
    losses = []
    temps = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.float().to(device), labels.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        losses.append(loss.item())
        temps.append(avg(model_temps(model)))
        optimizer.step() 
        if temp_schedule:
            temp_schedule.step()
        if batch_idx % args.log_interval == 0:
            t = avg(model_temps(model))
            inputs_seen = batch_idx * len(inputs)
            inputs_tot = len(train_loader.dataset)
            pct = 100. * batch_idx / len(train_loader)
            log_train_step(model, epoch, inputs_seen, inputs_tot, pct, loss.item(), t)

    if metrics_writer:
        record_metrics(metrics_writer, epoch, 'train', loss=avg(losses),
            temp=avg(temps))


def test(args, model, device, test_loader, criterion, num_labels):
    conf_mat = np.zeros((num_labels, num_labels))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            outputs = []
            for _ in range(args.inference_passes):
                outputs.append(model(inputs))
            mean_output = torch.mean(torch.stack(outputs), dim=0)
            pred = mean_output.argmax(dim=1)
            test_loss += criterion(mean_output, labels).sum().item()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            conf_mat += confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy(), labels=range(num_labels))

    test_loss /= len(test_loader.dataset)
    log_test(test_loss, correct, len(test_loader.dataset), conf_mat)
    return test_loss, correct/len(test_loader.dataset)


def _temp_scheduler(temps, args):
    if args.temp_jang:
        N, r, limit = args.temp_step, args.temp_exp, args.temp_limit
        return JangScheduler(temps, N, r, limit)
    else:
        return ConstScheduler(temps, args.temp_const)


def run_model(model, optimizer, start_epoch, args, train_loader, test_loader, device):
    criterion = nn.CrossEntropyLoss()

    handlers = [logging.StreamHandler()]
    metrics_writer = None
    if not args.no_log:
        if not path.exists(args.metrics_dir):
            mkdir(args.metrics_dir)
        metrics_path = path.join(args.metrics_dir, args.experiment_name)
        metrics_writer = SummaryWriter(log_dir=metrics_path)
        if not path.exists(args.log_dir):
            mkdir(args.log_dir)
        log_file = path.join(args.log_dir, f'{args.experiment_name}.log')
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(handlers=handlers, format='%(message)s', level=logging.INFO)

    temp_schedule = None if args.deterministic else _temp_scheduler(model_temps(model, val_only=False), args)

    # labels should be a whole number from [0, num_classes - 1]
    num_labels = 10 #int(max(max(train_data.targets), max(test_data.targets))) + 1

    logging.info("Model Architecture: ", model)
    logging.info("Using device: ", device)
    logging.info("Normalize layer outputs?: ", args.normalize)

    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        if args.adjust_lr:
            adjust_lr(args.lr, epoch, optimizer)
        train(args, model, device, train_loader, optimizer, epoch, criterion,
                metrics_writer, temp_schedule)
        loss, acc = test(args, model, device, test_loader, criterion, num_labels)
        if not args.no_log:
            record_metrics(metrics_writer, epoch, 'test', loss=loss, accuracy=acc)
        if (epoch % 10) == 0 and not args.no_save:
            checkpoint(model, optimizer, epoch+1, args.experiment_name)

    if not args.no_save:
        torch.save(model.state_dict(),
                f'checkpoints/{args.experiment_name}_{args.epochs}.pt')

    if not args.no_log:
        metrics_writer.flush()
        metrics_writer.close()
