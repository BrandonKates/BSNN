from matplotlib import pyplot as plt
import argparse

def plot_log(log_file, out_file):
    with open(log_file, 'r') as f:
        losses, accuracies, temps = [], [], []
        for line in f.readlines():
            if 'Loss:' in line:
                losses.append(float(line[line.find('Loss:') + len('Loss:'):]))
            if 'Accuracy:' in line:
                accuracies.append(
                    float(line[line.find('Accuracy:') + 9:line.find('/')])
                )
            if 'Temp:' in line:
                temps.append(
                    float(line[line.find('Temp:') + 5:line.find('\n')])
                )

    plt.plot(losses, accuracies, temps)
    if out_file:
        plt.savefig(out_file+'.png')
    else:
        plt.figure()

    #plt.show()


parser = argparse.ArgumentParser(description='Plot stuff')

parser.add_argument('logfile', 
        type=str, 
        help='path to log file')

parser.add_argument('--save-image', '-s', 
        type=str,
        default='',
        help='filename if you want to save plot')


args = parser.parse_args()

plot_log(args.logfile, args.save_image)
