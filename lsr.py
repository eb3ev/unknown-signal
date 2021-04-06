import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

available_args = [
        "--plot",
        "--equation",
        "--verbose"
]

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

def add_polynomial_term(X, x, degree):
    return np.column_stack((X, np.power(x, degree)))

def add_bias(X):
    return np.column_stack((np.ones(X.shape), X))

def fit_wh(X, Y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def square_err(y, y_h):
    return np.sum((y - y_h)**2)

def fit_line_of_best_fit(xs, model, *args):
    new_xs = np.linspace(xs.min(), xs.max(), len(xs))
    new_ys = model(new_xs, *args)
    return new_xs, new_ys

models = [
        lambda x, a, b: a + b*x, # 0: Linear
        lambda x, a, b, c: a + b*x + c*x**2, # 1: Quadratic
        lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3, # 2: Cubic
        lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4, # 3: Quartic
        lambda x, a, b, c, d, e, f: a + b*x + c*x**2 + d*x**3 + e*x**4 + a*x**5, # 4: Quintic
        lambda x, a, b: a + b*np.exp(x), # 6: Exponential
        lambda x, a, b: a + b*np.sin(x), # 7: Sine
        lambda x, a, b: a + b*np.cos(x), # 8: Cosine
]

modelNames = [
        "Linear",
        "Quadratic",
        "Cubic",
        "Quartic",
        "Quintic",
        "Exponential",
        "Sine",
        "Cosine"
]

modelEquations = [
        "a + bx", # 0: Linear
        "a + bx + cx^2", # 1: Quadratic
        "a + bx + cx^2 + dx^3", # 2: Cubic
        "a + bx + cx^2 + dx^3 + ex^4", # 3: Quartic
        "a + bx + cx^2 + dx^3 + ex^4 + fx^5", # 4: Quintic
        "a + be^x", # 5: Exponential
        "a + bsin(x)", # 6: Sine
        "a + bcos(x)" # 7: Cosine
]

def fit_linear(xs):
    xe = add_bias(xs)
    return xe

def fit_quadratic(xs):
    xe = fit_linear(xs)
    xe = add_polynomial_term(xe, xs, 2)
    return xe

def fit_cubic(xs):
    xe = fit_quadratic(xs)
    xe = add_polynomial_term(xe, xs, 3)
    return xe

def fit_quartic(xs):
    xe = fit_cubic(xs)
    xe = add_polynomial_term(xe, xs, 4)
    return xe

def fit_quintic(xs):
    xe = fit_quartic(xs)
    xe = add_polynomial_term(xe, xs, 5)
    return xe

def fit_exp(xs):
    xe = np.exp(xs)
    xe = add_bias(xe)
    return xe

def fit_sin(xs):
    xe = np.sin(xs)
    xe = add_bias(xe)
    return xe

def fit_cos(xs):
    xe = np.cos(xs)
    xe = add_bias(xe)
    return xe

feature_vectors = [
        fit_linear,
        fit_quadratic,
        fit_cubic,
        fit_quartic,
        fit_quintic,
        fit_exp,
        fit_sin,
        fit_cos
]

def fit_best(xs, ys, segment, isShowEquation, isVerbose):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        fit : List/array-like of the new x and y and y-hat values respectively in
            that order.
    """
    vals = []
    errs = []
    for i, model in enumerate(models):
        xe = feature_vectors[i](xs)
        wh = fit_wh(xe, ys)
        yh = model(xs, *wh)
        new_xs, new_ys = fit_line_of_best_fit(xs, model, *wh)
        vals.append([new_xs, new_ys, yh])
        err = square_err(ys, yh)
        errs.append(err)

    index = errs.index(min(errs))

    if isShowEquation:
        info = "Segment {}: {}".format(segment,modelNames[index])
        if isVerbose:
            info += " :: {}".format(modelEquations[index])
        print(info)

    fit = vals[index]
    return fit

def total_err(xs, ys, isPlot, isShowEquation, isVerbose):
    err = 0
    for i in range(num_line_segments(xs)):
        seg_x, seg_y = get_line_segment(xs, ys, i)

        new_xs, new_ys, yh = fit_best(seg_x, seg_y, i+1, isShowEquation, isVerbose)
        
        err += err + square_err(seg_y, yh) 
        if isPlot:
            plt.plot(new_xs, new_ys, c="#FF7A77")
    
    return err

def num_line_segments(data_points, segment_size = 20):
    assert len(data_points) % 20 == 0
    return len(data_points) // segment_size

# Order 1 means the first segment.
def get_line_segment(xs, ys, order):
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = num_line_segments(xs)

    start = len_data // num_segments * order
    end = start + (len_data // num_segments)

    return xs[start:end], ys[start:end]

def seperate_opts_and_fnames(args):
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

def main(argv):
    opts, fnames = seperate_opts_and_fnames(argv)

    if "--plot" in opts:
        isPlot = True
    else:
        isPlot = False

    if "--equation" in opts:
        isShowEquation = True
    else:
        isShowEquation = False

    if "--verbose" in opts:
        isVerbose = True
    else:
        isVerbose = False

    for fname in fnames:
        xs, ys = load_points_from_file(fname)
        print(total_err(xs, ys, isPlot, isShowEquation, isVerbose))
        if isPlot:
            view_data_segments(xs, ys)

main(sys.argv[1:])
