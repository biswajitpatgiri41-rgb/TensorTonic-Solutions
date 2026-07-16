import numpy as np

def pearson_correlation(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    n = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / (n - 1)
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)

    corr = np.full(cov.shape, np.nan, dtype=float)
    np.divide(cov, denom, out=corr, where=denom != 0)

    idx = np.arange(len(std))
    corr[idx, idx] = np.where(std == 0, np.nan, 1.0)

    return corr