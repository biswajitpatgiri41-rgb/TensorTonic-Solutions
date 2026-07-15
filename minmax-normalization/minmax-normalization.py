import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        x_min = np.min(X)
        x_max = np.max(X)
    else:
        x_min = np.min(X, axis=axis, keepdims=True)
        x_max = np.max(X, axis=axis, keepdims=True)
    denom = np.maximum(x_max - x_min, eps)
    return (X - x_min) / denom