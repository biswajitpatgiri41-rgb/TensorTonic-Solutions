import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    b, h, w, _ = x.shape

    for _ in range(4):
        h -= 4
        w -= 4
        h //= 2
        w //= 2

    h -= 4
    w -= 4

    for _ in range(4):
        h *= 2
        w *= 2
        h -= 4
        w -= 4

    return np.zeros((b, h, w, num_classes), dtype=x.dtype)