import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    new_m = beta1 * m + (1.0 - beta1) * grad
    new_v = beta2 * v + (1.0 - beta2) * (grad ** 2)
    new_w = w - lr * weight_decay * w - lr * (new_m / (np.sqrt(new_v) + eps))

    return new_w, new_m, new_v