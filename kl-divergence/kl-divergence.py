import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q_stable = q + eps
    mask = p > 0
    return np.sum(p[mask] * np.log(p[mask] / q_stable[mask]))