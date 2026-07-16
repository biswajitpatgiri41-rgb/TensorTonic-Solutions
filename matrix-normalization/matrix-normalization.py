import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2:
            return None

        if axis not in (None, 0, 1):
            return None

        if norm_type == 'l2':
            norms = np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True))
        elif norm_type == 'l1':
            norms = np.sum(np.abs(arr), axis=axis, keepdims=True)
        elif norm_type == 'max':
            norms = np.max(np.abs(arr), axis=axis, keepdims=True)
        else:
            return None

        norms = np.where(norms == 0, 1, norms)
        return arr / norms
    except Exception:
        return None