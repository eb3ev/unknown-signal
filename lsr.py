import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()

def seperate_options_and_filenames(args):
    available_args = ["--plot"]
    fnames = []
    opts = []

    for arg in args:
        if arg in available_args:
            opts.append(arg)
        elif os.path.isfile(arg) and arg.endswith(".csv"):
            fnames.append(arg)
        else:
            print("Invalid Arg: " + arg)

    return opts, fnames


def calc_total_reconstruction_error(xs, ys):
    return 0

def main(argv):
    opts, fnames = seperate_options_and_filenames(argv)

    if "--plot" in opts:
        isPlot = True
    else:
        isPlot = False

    for fname in fnames:
        xs, ys = load_points_from_file(fname)
        if isPlot:
            view_data_segments(xs, ys)
        
        print(calc_total_reconstruction_error(xs, ys))

main(sys.argv[1:])
