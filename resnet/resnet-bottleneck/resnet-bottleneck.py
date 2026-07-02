import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    relu = lambda z: np.maximum(0, z)

    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = None if Ws is None else np.array(Ws)

    shortcut = x if Ws is None else x @ Ws

    out = relu(x @ W1)
    out = relu(out @ W2)
    out = out @ W3

    return relu(out + shortcut)