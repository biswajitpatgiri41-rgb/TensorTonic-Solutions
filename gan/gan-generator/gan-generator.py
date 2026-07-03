import numpy as np

def generator(z, W, b):
    """
    Computes the output of a simple GAN generator layer.

    Args:
        z (np.ndarray): Input noise of shape (batch_size, noise_dim)
        W (np.ndarray): Weight matrix of shape (noise_dim, output_dim)
        b (np.ndarray): Bias vector of shape (output_dim,)

    Returns:
        np.ndarray: Output of shape (batch_size, output_dim) with
                    tanh activation applied and values rounded to 4 decimals.
    """
    z = np.asarray(z, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    output = np.tanh(np.dot(z, W) + b)
    return np.round(output, 4)