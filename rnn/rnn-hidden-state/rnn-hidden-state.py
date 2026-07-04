import numpy as np

def init_hidden(batch_size, hidden_dim):
    return np.zeros((batch_size, hidden_dim), dtype=float)