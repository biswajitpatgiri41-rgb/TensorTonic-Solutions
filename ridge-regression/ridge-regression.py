import numpy as np

def ridge_regression(X, y, lam):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    d = X.shape[1]
    I = np.eye(d)
    w = np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)
    return w.tolist()