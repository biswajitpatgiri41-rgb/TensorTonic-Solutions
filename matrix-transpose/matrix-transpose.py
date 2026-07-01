import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    n, m = A.shape
    result = np.zeros((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            result[j, i] = A[i, j]
    return result