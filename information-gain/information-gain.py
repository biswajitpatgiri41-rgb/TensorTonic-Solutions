import numpy as np

def _entropy(y):
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    left = y[split_mask]
    right = y[~split_mask]

    n_left = left.size
    n_right = right.size
    n = y.size

    if n_left == 0 or n_right == 0:
        return 0.0

    h_parent = _entropy(y)
    h_left = _entropy(left)
    h_right = _entropy(right)

    weighted_entropy = (n_left / n) * h_left + (n_right / n) * h_right
    return float(h_parent - weighted_entropy)