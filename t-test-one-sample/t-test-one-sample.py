import numpy as np

def t_test_one_sample(x, mu0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_bar = np.mean(x)
    s = np.sqrt(np.sum((x - x_bar) ** 2) / (n - 1))
    return float((x_bar - mu0) / (s / np.sqrt(n)))