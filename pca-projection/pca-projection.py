import numpy as np

def pca_projection(X, k):
    X = np.asarray(X, dtype=float)
    X_centered = X - np.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:k]
    W = eigenvectors[:, idx]
    return (X_centered @ W).tolist()