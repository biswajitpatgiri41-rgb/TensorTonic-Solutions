import numpy as np

def clip_gradients(g, max_norm):
    g = np.asarray(g)
    if max_norm <= 0:
        return g.copy()
    norm = np.linalg.norm(g)
    if norm == 0 or norm <= max_norm:
        return g.copy()
    return g * (max_norm / norm)