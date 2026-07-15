import numpy as np

def manhattan_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.sum(np.abs(x - y)))