import numpy as np

def euclidean_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.sqrt(np.sum((x - y) ** 2)))