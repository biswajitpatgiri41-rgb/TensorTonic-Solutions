import numpy as np

def normalize_3d(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else np.zeros_like(v)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-10)