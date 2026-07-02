import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model), dtype=float)
    positions = np.arange(seq_len).reshape(-1, 1)
    i = np.arange(0, d_model, 2).reshape(1, -1)
    angles = positions / np.power(base, i / d_model)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])
    return pe