import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    B, N, D = patches.shape
    assert N == num_patches
    assert D == embed_dim

    if pos_embed is None:
        pos_embed = np.random.randn(1, N, D) * 0.02

    return patches + pos_embed