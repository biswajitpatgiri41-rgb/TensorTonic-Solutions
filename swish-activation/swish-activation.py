import numpy as np

def swish(x):
    x = np.asarray(x, dtype=float)
    z = np.clip(x, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return x * sigmoid