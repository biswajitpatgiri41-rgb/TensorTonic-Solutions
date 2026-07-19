import numpy as np

def one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=int)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if y.size == 0:
        if num_classes is None:
            num_classes = 0
        return np.zeros((0, num_classes), dtype=float)
    if np.any(y < 0):
        raise ValueError("Labels must be non-negative")
    if num_classes is None:
        num_classes = np.max(y) + 1
    if np.any(y >= num_classes):
        raise ValueError("Labels must be less than num_classes")
    Y = np.zeros((y.shape[0], num_classes), dtype=float)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y