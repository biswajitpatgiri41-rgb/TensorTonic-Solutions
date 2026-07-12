import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    scores = np.asarray(scores, dtype=float)
    t = scores.shape[-1]
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)
    masked_scores = scores.copy()
    masked_scores[..., mask] = mask_value
    return masked_scores