import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    S = np.dot(Z1, Z2.T) / temperature
    S_stable = S - np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S_stable)
    log_sum_exp = np.log(np.sum(exp_S, axis=1))
    pos = np.diag(S_stable)
    loss = -np.mean(pos - log_sum_exp)
    return loss