import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    return float(np.mean(-0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)))