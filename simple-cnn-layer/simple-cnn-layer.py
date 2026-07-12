import numpy as np

def conv2d(x, W, b):
    x = np.asarray(x, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    if x.ndim != 4 or W.ndim != 4 or b.ndim != 1:
        raise ValueError

    N, C_in, H, W_in = x.shape
    C_out, C_w, KH, KW = W.shape

    if C_in != C_w or b.shape[0] != C_out:
        raise ValueError

    H_out = H - KH + 1
    W_out = W_in - KW + 1

    if H_out <= 0 or W_out <= 0:
        raise ValueError

    out = np.empty((N, C_out, H_out, W_out), dtype=float)

    for n in range(N):
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c_out, i, j] = (
                        np.sum(x[n, :, i:i + KH, j:j + KW] * W[c_out]) + b[c_out]
                    )

    return out