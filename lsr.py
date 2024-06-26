import os
import sys
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

SEGMENT_SIZE = 20
K_FOLD = 10

available_args = [
        "--plot",
        "--no-cross-validation",
        "--random-k-fold",
        "-v",
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
    """Extends X with a column of x^degree to the right.
    Args:
        X : List/array-like of feature vectors.
        x : List/array-like of x co-ordinates.
        degree : The power to raise x.
    Returns:
        An polynomially extended feature vector
    """
    return np.column_stack((X, np.power(x, degree)))

def add_bias(X):
    """Extends X with a column of ones to the left.
    Args:
        X : List/array-like of feature vectors.
        x : List/array-like of x co-ordinates.
    Returns:
        A feature vector with a column of ones in the 
        first column.
    """
    return np.column_stack((np.ones(X.shape), X))

def fit_wh(X, Y):
    """Least squares regression.
    Args:
        X : List/array-like of x's feature vectors.
        Y : List/array-like of y co-ordinates.
    Returns:
        The coefficient vector of the least squares regression.
    """
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

def square_err(y, y_h):
    return np.sum((y - y_h)**2)

def fit_line_of_best_fit(xs, model, *args):
    """Least squares regression.
    Args:
        xs : List/array-like of x co-ordinates.
        model : The model to use for the line.
        *args : The coefficients to use.
    Returns:
        The List/array-like of x co-ordinates and a
        List/array-like of y co-ordinates to plot respectively
        in that order.
    """
    new_xs = np.linspace(xs.min(), xs.max(), len(xs) * 2)
    new_ys = model(new_xs, *args)
    return new_xs, new_ys

def fit_linear(xs):
    xe = add_bias(xs)
    return xe

def fit_polynomial(xs, order):
    assert order >= 2
    xe = fit_linear(xs)
    for i in range(2, order+1):
        xe = add_polynomial_term(xe, xs, i)
    return xe

def fit_exp(xs):
    xe = np.exp(xs)
    xe = add_bias(xe)
    return xe

def fit_trigonometry(xs, func):
    xe = func(xs)
    xe = add_bias(xe)
    return xe

# List of the functions of the models accompanied by their feature vector functions.
models = [
        [
            "Linear",
            fit_linear,
            lambda x, a, b: a + b*x
        ],    
        [
            "Quadratic",
            lambda x: fit_polynomial(x, 2),
            lambda x, a, b, c: a + b*x + c*x**2
        ],
        [
            "Cubic",
            lambda x: fit_polynomial(x, 3),
            lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3
        ],
        [
            "Quartic",
            lambda x: fit_polynomial(x, 4),
            lambda x, a, b, c, d, e: a + b*x + c*x**2 + d*x**3 + e*x**4
        ],
        [
            "Quintic",
            lambda x: fit_polynomial(x, 5),
            lambda x, a, b, c, d, e, f: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5
        ],
        [
            "6th Order Polynomial",
            lambda x: fit_polynomial(x, 6),
            lambda x, a, b, c, d, e, f, g: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6

        ],
        [
            "7th Order Polynomial",
            lambda x: fit_polynomial(x, 7),
            lambda x, a, b, c, d, e, f, g, h: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7
        ],
        [
            "8th Order Polynomial",
            lambda x: fit_polynomial(x, 8),
            lambda x, a, b, c, d, e, f, g, h, i: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8
        ],
        [
            "9th Order",
            lambda x: fit_polynomial(x, 9),
            lambda x, a, b, c, d, e, f, g, h, i, j: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9
        ],
        [
            "10th Order",
            lambda x: fit_polynomial(x, 10),
            lambda x, a, b, c, d, e, f, g, h, i, j, k: a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9 + k*x**10
        ],
        [
            "Exponential",
            fit_exp,
            lambda x, a, b: a + b*np.exp(x)
        ],
        [
            "Sinusoidal",
            lambda x: fit_trigonometry(x, np.sin),
            lambda x, a, b: a + b*np.sin(x)
        ],
        [
            "Cosinusoidal",
            lambda x: fit_trigonometry(x, np.cos),
            lambda x, a, b: a + b*np.cos(x)
        ],
]

# Comment out models to not not include in training.
model_whitelist = [
        0, # Linear
        # 1, # Quadratic
        2, # Cubic
        # 3, # Quadratic
        # 4, # Quintic
        # 5, # 6th Order Polynomial
        # 6, # 7th Order Polynomial
        # 7, # 8th Order Polynomoal
        # 8, # 9th Order Polynomial
        # 9, # 10th Order Polynomial
        # 10, # Exponential
        11, # Sine
        # 12, # Cosine
]

def k_fold_parts(xs, ys, k = 10, randomize = False):
    assert len(xs) >= k
    assert k >= 2

    if k == 1:
        return [xs], [ys]

    len_data = len(xs)
    len_fold = len_data // k

    x_parts = [np.array([])]
    y_parts = [np.array([])]
    for i in range(k - 1):
        x_parts.append(np.array([]))
        y_parts.append(np.array([]))

    if randomize:
        part = 0
        indices = list(range(len_data))
        random.shuffle(indices)
        for i in indices:
            x_parts[part] = np.append(x_parts[part], xs[i])
            y_parts[part] = np.append(y_parts[part], ys[i])
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

def cross_validation(xs, ys, feature_vector, model, k = K_FOLD, rand_k_fold = False):
    err = 0
    x_parts, y_parts = k_fold_parts(xs, ys, k, rand_k_fold)
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
        try:
            wh = fit_wh(xe, train_ys)
            yh = model(test_xs, *wh)
            err += square_err(test_ys, yh)
        except Exception:
            k -= 1

    return err / k

def fit_best(xs, ys, segment, use_cross_validation = True, k = K_FOLD, rand_k_fold = False, verbose = False):
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
    indices = []
    for i, model in enumerate(models):
        if i in model_whitelist:
            try:
                xe = model[1](xs)
                wh = fit_wh(xe, ys)
                yh = model[2](xs, *wh)
                new_xs, new_ys = fit_line_of_best_fit(xs, model[2], *wh)
                vals.append([new_xs, new_ys, yh])
                if use_cross_validation:
                    err = cross_validation(xs, ys, model[1], model[2], k , rand_k_fold)
                else:
                    err = square_err(ys, yh)
                errs.append(err)
                indices.append(i)
            except Exception:
                pass

    index = errs.index(min(errs))

    if verbose:
        info = "Segment {}: {}".format(segment, models[indices[index]][0])
        print(info)

    fit = vals[index]
    return fit

def total_err(xs, ys, use_cross_validation = True, k = K_FOLD, rand_k_fold = False, plot = False, verbose = False):
    err = 0
    for i in range(num_line_segments(xs, SEGMENT_SIZE)):
        seg_x, seg_y = get_line_segment(xs, ys, i)

        new_xs, new_ys, yh = fit_best(seg_x, seg_y, i+1, use_cross_validation, k, rand_k_fold, verbose)
        
        err += square_err(seg_y, yh) 
        if plot:
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
        if arg in available_args or arg.startswith("-k="):
            opts.append(arg)
        elif os.path.isfile(arg) and arg.endswith(".csv"):
            fnames.append(arg)
        else:
            sys.exit("Invalid Arg: " + arg)

    return opts, fnames

def main(argv):
    opts, fnames = seperate_opts_and_fnames(argv)

    plot, use_cross_validation, k, rand_k_fold, verbose = False, True, K_FOLD, False, False

    for opt in opts:
        if opt == "--plot":
            plot = True
        elif opt == "-v":
            verbose = True
        elif opt == "--no-cross-validation":
            use_cross_validation = False
        elif opt.startswith("-k="):
            k = int(opt[3:])
        elif opt == "--random-k-fold":
            rand_k_fold = True

    for fname in fnames:
        xs, ys = load_points_from_file(fname)
        print(total_err(xs, ys, use_cross_validation, k, rand_k_fold, plot, verbose))
        if plot:
            view_data_segments(xs, ys)

main(sys.argv[1:])
