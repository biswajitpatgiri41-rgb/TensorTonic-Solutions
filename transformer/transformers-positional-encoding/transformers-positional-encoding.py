import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings as described in
    "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        seq_length (int): Length of the sequence.
        d_model (int): Embedding dimension.

    Returns:
        np.ndarray: Positional encoding matrix of shape (seq_length, d_model)
                    with dtype float64.
    """
    
    positions = np.arange(seq_length, dtype=np.float64).reshape(-1, 1)

    
    dims = np.arange(0, d_model, 2, dtype=np.float64)

    
    div_term = np.exp(dims * (-np.log(10000.0) / d_model))

    
    pe = np.zeros((seq_length, d_model), dtype=np.float64)

    
    pe[:, 0::2] = np.sin(positions * div_term)

    
    pe[:, 1::2] = np.cos(positions * div_term)

    return pe