import numpy as np

def dot_product(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Inputs must have the same length")
    return float(np.dot(x, y))