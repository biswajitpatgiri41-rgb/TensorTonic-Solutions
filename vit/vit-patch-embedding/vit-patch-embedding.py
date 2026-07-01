import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    B, H, W, C = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0

    h_patches = H // P
    w_patches = W // P
    N = h_patches * w_patches
    patch_dim = P * P * C

    patches = image.reshape(B, h_patches, P, w_patches, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, N, patch_dim)

    if W_proj is None:
        W_proj = np.random.randn(patch_dim, embed_dim) * 0.02

    embeddings = patches @ W_proj
    return embeddings