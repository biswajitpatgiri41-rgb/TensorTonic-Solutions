import numpy as np

def rnn_step_backward(dh, cache):
    x_t, h_prev, h_t, W, U, b = cache

    dh = np.array(dh)
    x_t = np.array(x_t)
    h_prev = np.array(h_prev)
    h_t = np.array(h_t)
    W = np.array(W)
    U = np.array(U)
    b = np.array(b)

    dz = dh * (1 - h_t * h_t)
    dx_t = W.T @ dz
    dh_prev = U.T @ dz
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz

    return dx_t, dh_prev, dW, dU, db