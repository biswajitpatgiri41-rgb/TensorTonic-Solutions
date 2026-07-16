import numpy as np
import math

def gelu(x):
    x = np.asarray(x, dtype=float)
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))