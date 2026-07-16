import numpy as np

def calculate_eigenvalues(matrix):
    try:
        matrix = np.asarray(matrix)
    except Exception:
        return None

    if matrix.ndim != 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return np.array([], dtype=complex)

    try:
        eigenvalues = np.linalg.eigvals(matrix)
        order = np.lexsort((eigenvalues.imag, eigenvalues.real))
        return eigenvalues[order]
    except Exception:
        return None