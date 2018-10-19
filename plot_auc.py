#!/usr/bin/python
"""
Author: Liang Zhao

For example, the log file contrains the following log
   Avg Run Time (ms/batch): 14.034 AUC: 0.789 max AUC: 0.806

To use this script to generate plot for AUC:
   python plot_auc.py -i loss_log.txt -o figure.png AUC
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
    print('Min err_g: train = %f @ %d' % (max(x[:,2]), x[numpy.argmax(x[:,2]),0]))

    m = len(test_keys) + 1
    for i in range(1, m):
        if (x_test.shape[0] > 0):
            pyplot.plot(
                range(len(x_test)), #x[:, 0],
                x_test[:, i],
                #color=cm.jet(1.0 - 1.0 * (i - 1) / (2 * m)),
                label=test_keys[i - 1])
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
