from matplotlib import pyplot as plt
import numpy as np
import argparse
import os

def plot_log(log_files, out_file, display, window_size):
    for log_file in log_files:
        with open(log_file, 'r') as f:
            losses, accuracies = [], []
            for line in f.readlines():
                if 'Loss:' in line:
                    losses.append(float(line[line.find('Loss:') + len('Loss:'):line.find('\tTemp:')]))
                if 'Accuracy:' in line:
                    accuracies.append(float(line[line.find('Accuracy:')
                        + len('Accuracy:'):line.find('/')])/10000)


        if window_size > 1 and window_size <= len(losses):
            ll = len(losses)
            norm = 1/window_size
            avg = sum(losses[:window_size])*norm
            avgs = [avg]
            for rem_i, add_i in zip(range(ll-window_size), range(window_size,ll)):
                rem , add = losses[rem_i], losses[add_i]
                avg += (norm * (add-rem))
                avgs.append(avg)

            losses = avgs

        #plt.plot(losses)
        plt.plot(accuracies)

    plt.legend(list(map(lambda f: os.path.basename(f), log_files)))
    plt.axes().set_title('Accuracy over Epochs')
    plt.axes().set_xlabel('Epochs')
    plt.axes().set_ylabel('Test Accuracy')

    if out_file:
        plt.savefig(out_file+'.png')
    if display:
        plt.show()


parser = argparse.ArgumentParser(description='Plot stuff')

parser.add_argument('logfiles', 
        nargs='+', 
        help='path to log file')

parser.add_argument('--output-image', '-o', 
        type=str,
        default='',
        help='filename if you want to save plot')

parser.add_argument('--no-display', '-n',
        action='store_true',
        help='provide if you want to only save plot')

parser.add_argument('--smooth', '-s', 
        type=int,
        default=1,
        help='window size to smooth output in')


args = parser.parse_args()
plot_log(args.logfiles,args.output_image,not args.no_display,args.smooth)
