import numpy as np

def local_response_normalization(
    x: np.ndarray,
    k: float = 2,
    n: int = 5,
    alpha: float = 1e-4,
    beta: float = 0.75
) -> np.ndarray:
    
    if x.ndim != 4:
        raise ValueError("Input must have shape (B, H, W, C).")

    B, H, W, C = x.shape
    squared = x ** 2
    output = np.empty_like(x, dtype=x.dtype)

    half = n // 2

    for i in range(C):
        start = max(0, i - half)
        end = min(C, i + half + 1)

        channel_sum = np.sum(squared[..., start:end], axis=-1)
        denom = (k + alpha * channel_sum) ** beta
        output[..., i] = x[..., i] / denom

    return output