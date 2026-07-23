import numpy as np

def vector_norm_3d(v):
    v = np.asarray(v)
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v**2)))
    elif v.ndim == 2:
        return np.sqrt(np.sum(v**2, axis=1))
    else:
        raise ValueError("Input must be a 3D vector or a batch of 3D vectors.")