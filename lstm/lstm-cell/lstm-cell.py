import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    is_1d = x_t.ndim == 1

    if is_1d:
        x_t = x_t[np.newaxis, :]
        h_prev = h_prev[np.newaxis, :]
        C_prev = C_prev[np.newaxis, :]

    concat = np.concatenate((h_prev, x_t), axis=-1)

    f_t = sigmoid(concat @ W_f.T + b_f)
    i_t = sigmoid(concat @ W_i.T + b_i)
    C_tilde = np.tanh(concat @ W_c.T + b_c)
    o_t = sigmoid(concat @ W_o.T + b_o)

    C_t = f_t * C_prev + i_t * C_tilde
    h_t = o_t * np.tanh(C_t)

    if is_1d:
        return h_t[0], C_t[0]

    return h_t, C_t