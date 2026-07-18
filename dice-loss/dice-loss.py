import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.asarray(p, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    intersection = np.sum(p * y)
    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)
    return 1.0 - dice