import numpy as np

def generator(z, W, b):
    z = np.asarray(z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.round(np.tanh(np.dot(z, W) + b), 4)

def discriminator(x, W):
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    logits = np.dot(x, W)
    probabilities = 1 / (1 + np.exp(-logits))
    return np.round(probabilities, 4)