#!/usr/bin/python
"""
Author: Liang Zhao

For example, the log file contrains the following log
   Avg Run Time (ms/batch): 14.034 AUC: 0.789 max AUC: 0.806

To use this script to generate plot for AUC:
   python cmp_auc.py -i loss_log_1 loss_log_2 -o figure.png AUC

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

def plot_curve(keys, inputfile, label='AUC'):
    """Plot a curve from log based on keys .

    :param keys: a list of strings to be plotted, e.g. AUC 
    :param inputfile: a file object for input
    :param outputfile: a file object for output
    :return: None
    """
    test_pattern = r"\(ms\/batch\)\: ([0-9\.]*)"

    if not keys:
        test_keys = ['AUC']
        
    for k in test_keys:
        test_pattern += r".*?%s: ([0-9e\-\.]*)" % k
    test_data = []
    compiled_test_pattern = re.compile(test_pattern)
    for line in inputfile:
        found_test = compiled_test_pattern.search(line)
        if found_test:
            test_data.append([float(x) for x in found_test.groups()])
    x_test = numpy.array(test_data)
    if x_test.shape[0] <= 0:
        sys.stderr.write("No data to plot. Exiting!\n")
        return
    print('Max AUC = %f @ %d' % (max(x_test[:,1]), numpy.argmax(x_test[:,1])))

    m = len(test_keys) + 1
    for i in range(1, m):
        if (x_test.shape[0] > 0):
            pyplot.plot(
                range(len(x_test)), 
                x_test[:, i],
                label=label)

def plot_multi_curves(keys, inputfiles, outputfile, format='png',
               show_fig=False):
    labels = ['AUC 1', 'AUC 2']

    for i in range(len(inputfiles)): 
      inputfile = open(inputfiles[i])
      plot_curve(keys, inputfile, labels[i])
      inputfile.close()

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
        '--inputs',
        nargs='*',
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
    format = args.format
    if args.output:
        outputfile = open(args.output, 'wb')
        if not format:
            format = os.path.splitext(args.output)[1]
            if not format:
                format = 'png'
    else:
        outputfile = sys.stdout

    #pdb.set_trace()
    inputfiles = [inputfile+'.txt' for inputfile in args.inputs]

    plot_multi_curves(keys, inputfiles, outputfile, format)
    outputfile.close()


if __name__ == "__main__":
    main(sys.argv[1:])
