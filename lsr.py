import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

SEGMENT_SIZE = 20

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

def view_data_segments(xs, ys, segment_size = SEGMENT_SIZE):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % segment_size == 0
    len_data = len(xs)
    num_segments = len_data // segment_size
    colour = np.concatenate([[i] * segment_size for i in range(num_segments)])
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

def fit_nine(xs):
    xe = fit_quintic(xs)
    xe = add_polynomial_term(xe, xs, 6)
    xe = add_polynomial_term(xe, xs, 7)
    xe = add_polynomial_term(xe, xs, 8)
    xe = add_polynomial_term(xe, xs, 9)
    return xe


models = [
        # Lambda Equations                                                        Feature Vectors   Name           Equation
        [lambda x, a, b: a + b*x,                                                 fit_linear,       "Linear",      "a + bx"                            ],
        # [lambda x, a, b, c: a + b*x + c*x**2,                                     fit_quadratic,    "Quadratic",   "a + bx + cx^2"                     ],
        [lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3,                         fit_cubic,        "Cubic",       "a + bx + cx^2 + dx^3"              ],
        # [lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4,             fit_quartic,      "Quartic",     "a + bx + cx^2 + dx^3 + ex^4"       ],
        # [lambda x, a, b, c, d, e, f: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5, fit_quintic,      "Quintic",     "a + bx + cx^2 + dx^3 + ex^4 + fx^5"],
        # [lambda x, a, b, c, d, e, f, g, h, i, j: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9, fit_nine, "nine", "nine"],
        # [lambda x, a, b: a + b*np.exp(x),                                         fit_exp,          "Exponential", "a + be^x"                          ],
        [lambda x, a, b: a + b*np.sin(x),                                         fit_sin,          "Sine",        "a + bsin(x)"                       ],
        # [lambda x, a, b: a + b*np.cos(x),                                         fit_cos,          "Cosine",      "a + bcos(x)"                       ],
]

def k_fold_parts(xs, ys, k = 10, randomize = False):
    assert len(xs) >= k

    len_data = len(xs)
    len_fold = len_data // k

    x_parts = [np.array([])]
    y_parts = [np.array([])]
    for i in range(k - 1):
        x_parts.append(np.array([]))
        y_parts.append(np.array([]))


    if randomize:
        part = 0
        used_indices = []
        for _ in range(len(xs)):
            index = random.randint(0, len_data - 1)
            for _ in used_indices:
                if not index in used_indices:
                    break
                else:
                    index = random.randint(0, len_data - 1)
                    used_indices.append(index)

            x_parts[part] = np.append(x_parts[part], xs[index])
            y_parts[part] = np.append(y_parts[part], ys[index])
            len_data -= 1
            if len(x_parts[part]) >= len_fold and part < k - 1:
                part += 1
    else:
        start = 0
        end = start + len_fold
        
        for i in range(k):
            if i == k - 1:
                end = len_data

            x_parts[i] = np.append(x_parts[i], xs[start:end])
            y_parts[i] = np.append(y_parts[i], ys[start:end])

            start += len_fold
            end += len_fold

    return x_parts, y_parts

def cross_validation(xs, ys, feature_vector, model):
    k = 10
    err = 0
    x_parts, y_parts = k_fold_parts(xs, ys, k, False)

    for i in range(k):
        train_xs, train_ys, test_xs, test_ys = np.array([]), np.array([]), np.array([]), np.array([])
        for j, part in enumerate(x_parts):
            if i == j:
                test_xs = np.append(test_xs, part)
            else:
                train_xs = np.append(train_xs, part)
        for j, part in enumerate(y_parts):
            if i == j:
                test_ys = np.append(test_ys, part)
            else:
                train_ys = np.append(train_ys, part)

        xe = feature_vector(train_xs)
        wh = fit_wh(xe, train_ys)
        yh = model(test_xs, *wh)

        err += square_err(test_ys, yh)

    return err / k

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
    use_cross_validation = True
    for i, model in enumerate(models):
        xe = model[1](xs)
        wh = fit_wh(xe, ys)
        yh = model[0](xs, *wh)
        new_xs, new_ys = fit_line_of_best_fit(xs, model[0], *wh)
        vals.append([new_xs, new_ys, yh])
        if use_cross_validation:
            err = cross_validation(xs, ys, model[1], model[0])
        else:
            err = square_err(ys, yh)
        errs.append(err)

    index = errs.index(min(errs))

    if isShowEquation:
        info = "Segment {}: {}".format(segment,models[index][2])
        if isVerbose:
            info += " :: {}".format(models[index][3])
        print(info)

    fit = vals[index]
    return fit

def total_err(xs, ys, isPlot, isShowEquation, isVerbose):
    err = 0
    for i in range(num_line_segments(xs, SEGMENT_SIZE)):
        seg_x, seg_y = get_line_segment(xs, ys, i)

        new_xs, new_ys, yh = fit_best(seg_x, seg_y, i+1, isShowEquation, isVerbose)
        
        err += square_err(seg_y, yh) 
        if isPlot:
            plt.plot(new_xs, new_ys, c="#FF7A77")
    
    return err

def num_line_segments(data_points, segment_size):
    assert len(data_points) % segment_size == 0
    return len(data_points) // segment_size

# Order 1 means the first segment.
def get_line_segment(xs, ys, order, segment_size = SEGMENT_SIZE):
    assert len(xs) == len(ys)
    assert len(xs) % segment_size == 0
    len_data = len(xs)
    num_segments = num_line_segments(xs, segment_size)

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

    if "-e" in opts:
        isShowEquation = True
    else:
        isShowEquation = False

    if "-v" in opts:
        isVerbose = True
    else:
        isVerbose = False

    isShowEquation = True

    for fname in fnames:
        xs, ys = load_points_from_file(fname)
        print(total_err(xs, ys, isPlot, isShowEquation, isVerbose))
        if isPlot:
            view_data_segments(xs, ys)

main(sys.argv[1:])
