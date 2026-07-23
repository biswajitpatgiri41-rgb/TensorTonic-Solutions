import numpy as np

def angle_between_3d(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-10 or nw < 1e-10:
        return np.nan
    cos_theta = np.dot(v, w) / (nv * nw)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.arccos(cos_theta))