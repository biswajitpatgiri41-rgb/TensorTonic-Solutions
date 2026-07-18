import numpy as np

def geometric_pmf_mean(k, p):
    k = np.array(k)
    pmf = ((1 - p) ** (k - 1)) * p
    mean = 1.0 / p
    return pmf, float(mean)
    