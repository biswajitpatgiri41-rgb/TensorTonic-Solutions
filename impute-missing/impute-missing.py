import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float).copy()

    if X.ndim == 1:
        mask = ~np.isnan(X)
        if np.any(mask):
            if strategy == 'mean':
                value = np.mean(X[mask])
            elif strategy == 'median':
                value = np.median(X[mask])
            else:
                raise ValueError("strategy must be 'mean' or 'median'")
        else:
            value = 0.0
        X[np.isnan(X)] = value
        return X

    if strategy not in ('mean', 'median'):
        raise ValueError("strategy must be 'mean' or 'median'")

    for j in range(X.shape[1]):
        mask = ~np.isnan(X[:, j])
        if np.any(mask):
            if strategy == 'mean':
                value = np.mean(X[mask, j])
            else:
                value = np.median(X[mask, j])
        else:
            value = 0.0
        X[np.isnan(X[:, j]), j] = value

    return X