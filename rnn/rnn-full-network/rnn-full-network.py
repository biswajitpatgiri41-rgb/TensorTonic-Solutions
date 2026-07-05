import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim

        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        batch_size, T, _ = X.shape

        if h_0 is None:
            h = np.zeros((batch_size, self.hidden_dim))
        else:
            h = h_0

        y_seq = []
        for t in range(T):
            x_t = X[:, t, :]
            h = np.tanh(x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            y = h @ self.W_hy.T + self.b_y
            y_seq.append(y)

        y_seq = np.stack(y_seq, axis=1)
        return y_seq, h