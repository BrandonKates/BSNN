from matplotlib import pyplot as plt
import numpy as np
import argparse

def plot_log(log_file, out_file, display):
    with open(log_file, 'r') as f:
        losses, temps = [], []
        for line in f.readlines():
            if 'Loss:' in line:
                losses.append(float(line[line.find('Loss:') + len('Loss:'):]))
            if 'Temp:' in line:
                temps.append(
                    float(line[line.find('Temp:') + 5:line.find('\n')])
                )

    plt.plot(losses)
    plt.plot(temps)

    if out_file:
        plt.savefig(out_file+'.png')
    if display:
        plt.figure()
        plt.show()


parser = argparse.ArgumentParser(description='Plot stuff')

parser.add_argument('logfile', 
        type=str, 
        help='path to log file')

parser.add_argument('--save-image', '-s', 
        type=str,
        default='',
        help='filename if you want to save plot')

parser.add_argument('--no-display', '-n',
        action='store_true',
        help='provide if you want to only save plot')


args = parser.parse_args()

plot_log(args.logfile, args.save_image, display=not args.no_display)
