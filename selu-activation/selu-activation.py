import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    x = np.asarray(x, dtype=float)
    y = np.where(x > 0, lam * x, lam * alpha * (np.exp(x) - 1))
    return np.round(y, 4).tolist()