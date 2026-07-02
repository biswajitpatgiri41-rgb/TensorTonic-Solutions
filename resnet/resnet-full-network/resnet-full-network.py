import numpy as np

def relu(x):
    return np.maximum(0, x)

def basic_block(x, W1, W2, W_proj=None):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    if W_proj is not None:
        W_proj = np.array(W_proj)
        identity = x @ W_proj
    else:
        identity = x
    out = relu(x @ W1)
    out = out @ W2
    out = relu(out + identity)
    return out

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    x = np.array(x)
    conv1 = np.array(conv1)
    fc = np.array(fc)
    out = relu(x @ conv1)
    out = basic_block(out, W1_b1, W2_b1)
    out = basic_block(out, W1_b2, W2_b2, Ws_b2)
    out = out @ fc
    return out