import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x)
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1).")
    if p == 0.0:
        return x.copy(), np.ones_like(x, dtype=float)
    keep_prob = 1.0 - p
    rand = rng.random(x.shape) if rng is not None else np.random.random(x.shape)
    mask = (rand < keep_prob).astype(float)
    pattern = mask / keep_prob
    output = x * pattern
    return output, pattern