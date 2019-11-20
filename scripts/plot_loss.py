from matplotlib import pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Plot stuff')

parser.add_argument('--log-file', '-lf', type=str, required=True, 
		                    help='which log file do you want to plot?')
parser.add_argument('--loss', action='store_true', help='plot training loss?')
parser.add_argument('--accuracy', action='store_true', help='plot testing accuracy?')


args = parser.parse_args()

def plot_loss(filename):
    f = open(filename)
    losses = []
    keyword = 'Loss:'
    for line in f.readlines():
        if keyword in line:
            losses.append(float(line[line.find(keyword) + len(keyword):]))
    plt.plot(losses)
    plt.title('Loss')
    plt.savefig('loss.png')

def plot_accuracy(filename):
    f = open(filename)
    accuracies = []
    keyword = 'Accuracy:'

    for line in f.readlines():
        if keyword in line:
            accuracies.append(float(line[line.find(keyword) + len(keyword):line.find('/')]))
    plt.plot(accuracies)
    plt.title('Accuracy')
    plt.savefig('accuracy.png')

if args.loss:
    plot_loss(args.log_file)
    plt.figure()

if args.accuracy:
    plot_accuracy(args.log_file)

plt.show()

