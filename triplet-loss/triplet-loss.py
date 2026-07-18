import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    anchor = np.asarray(anchor, dtype=np.float64)
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)

    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)

    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)
    loss = np.maximum(0.0, d_ap - d_an + margin)
    return float(np.mean(loss))