import numpy as np

def percentiles(x, q):
    x = np.asarray(x)
    q = np.asarray(q)
    return np.percentile(x, q, method="linear")