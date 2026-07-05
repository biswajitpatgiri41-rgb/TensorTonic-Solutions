import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    gradient = 1.0
    norms = []
    for _ in range(T):
        norms.append(float(gradient))
        gradient *= spectral_norm
    return norms