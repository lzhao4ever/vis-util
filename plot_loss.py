#!/usr/bin/python
"""
Author: Liang Zhao

For example, the log file contrains the following log
    Loss: [109/110] err_d: 0.473 err_g: 7.761 err_d_real: 0.218 err_d_fake: 0.255 err_g_bce: 2.380 err_g_l1l: 0.104 err_g_enc: 0.192

To use this script to generate plot for error:
   python plot_loss.py -i loss_log.txt -o figure.png err 
"""

import sys
import matplotlib
# the following line is added immediately after import matplotlib
# and before import pylot. The purpose is to ensure the plotting
# works even under remote login (i.e. headless display)
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as pyplot
import numpy
import argparse
import re
import os
import pdb

def plot_curve(keys, inputfile, outputfile, format='png',
                      show_fig=False):
    """Plot curves from log and save to outputfile.

    :param keys: a list of strings to be plotted, e.g. AUC 
    :param inputfile: a file object for input
    :param outputfile: a file object for output
    :return: None
    """
    pass_pattern = r"Loss\: \[([0-9]*)"
    test_pattern = r"\(ms\/batch\)\: ([0-9\.]*)"

    if not keys:
        keys = ['err_d', 'err_g']
        test_keys = ['AUC', 'max AUC']
        
    for k in keys:
        pass_pattern += r".*?%s: ([0-9e\-\.]*)" % k
    for k in test_keys:
        test_pattern += r".*?%s: ([0-9e\-\.]*)" % k
    data = []
    test_data = []
    compiled_pattern = re.compile(pass_pattern)
    compiled_test_pattern = re.compile(test_pattern)
    for line in inputfile:
        found = compiled_pattern.search(line)
        found_test = compiled_test_pattern.search(line)
        if found:
            #pdb.set_trace()
            data.append([float(x) for x in found.groups()])
        if found_test:
            #pdb.set_trace()
            test_data.append([float(x) for x in found_test.groups()])
    x = numpy.array(data)
    x_test = numpy.array(test_data)
    if x_test.shape[0] <= 0:
        sys.stderr.write("No data to plot. Exiting!\n")
        return
    print('Max AUC: train = %f @ %d' % (max(x_test[:,1]), numpy.argmax(x_test[:,1])))
    print('Min err_d: train = %f @ %d' % (max(x[:,1]), x[numpy.argmax(x[:,1]),0]))

    m = len(keys) + 1
    for i in range(1, m):
        pyplot.plot(
            x[:, 0],
            x[:, i],
            #color=cm.jet(1.0 * (i - 1) / (2 * m)),
            label=keys[i - 1])
    pyplot.xlabel('epoch')
    pyplot.legend(loc='best')
    if show_fig:
        pyplot.show()
    pyplot.savefig(outputfile, bbox_inches='tight')
    pyplot.clf()


def main(argv):
    """
    main method of plotting curves.
    """
    cmdparser = argparse.ArgumentParser(
        "Plot training and testing curves from log file.")
    cmdparser.add_argument(
        'key', nargs='*', help='keys of scores to plot, the default is AUC')
    cmdparser.add_argument(
        '-i',
        '--input',
        help='input filename of log, '
        'default will be standard input')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output filename of figure, '
        'default will be standard output')
    cmdparser.add_argument('--format', help='figure format(png|pdf|ps|eps|svg)')
    args = cmdparser.parse_args(argv)
    keys = args.key
    if args.input:
        inputfile = open(args.input)
    else:
        inputfile = sys.stdin
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout
    plot_curve(keys, inputfile, outputfile, format)
    inputfile.close()
    outputfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
