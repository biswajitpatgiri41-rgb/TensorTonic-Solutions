import numpy as np

def covariance_matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    X_centered = X - np.mean(X, axis=0)
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)