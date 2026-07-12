import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    if x.ndim == 2:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return x_hat * gamma + beta

    if x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        return x_hat * gamma + beta

    raise ValueError("Input must have shape (N,D) or (N,C,H,W)")