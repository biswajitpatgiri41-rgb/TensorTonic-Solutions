import numpy as np

def relu(x):
    return np.maximum(0, x)

def identity_block(x, W1, W2):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    identity = x
    out = x @ W1.T
    out = relu(out)
    out = out @ W2.T
    out = out + identity
    out = relu(out)
    return out