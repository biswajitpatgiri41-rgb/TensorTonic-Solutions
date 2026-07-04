import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    h = h_0
    h_list = []
    T = X.shape[1]

    for t in range(T):
        x = X[:, t, :]
        h = np.tanh(x @ W_xh.T + h @ W_hh.T + b_h)
        h_list.append(h)

    hidden_states = np.stack(h_list, axis=1)
    h_final = h

    return hidden_states, h_final