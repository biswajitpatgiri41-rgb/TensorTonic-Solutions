import numpy as np

def global_avg_pool(x):
    x = np.asarray(x)
    if x.ndim == 3:
        return x.mean(axis=(1, 2), dtype=float)
    if x.ndim == 4:
        return x.mean(axis=(2, 3), dtype=float)
    raise ValueError("Input must have shape (C, H, W) or (N, C, H, W)")