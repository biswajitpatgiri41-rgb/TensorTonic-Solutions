import numpy as np

def matrix_inverse(A):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    if abs(np.linalg.det(A)) < 1e-10:
        return None
    return np.linalg.inv(A)