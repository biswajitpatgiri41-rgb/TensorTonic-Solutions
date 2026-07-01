import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq: np.ndarray = None, Wk: np.ndarray = None, Wv: np.ndarray = None,
                      Wo: np.ndarray = None, W1: np.ndarray = None, W2: np.ndarray = None) -> np.ndarray:
    B, N, D = x.shape
    assert D == embed_dim
    assert D % num_heads == 0
    head_dim = D // num_heads
    eps = 1e-6

    def layer_norm(t):
        mean = t.mean(axis=-1, keepdims=True)
        var = t.var(axis=-1, keepdims=True)
        return (t - mean) / np.sqrt(var + eps)

    if Wq is None:
        Wq = np.random.randn(D, D) * 0.02
    if Wk is None:
        Wk = np.random.randn(D, D) * 0.02
    if Wv is None:
        Wv = np.random.randn(D, D) * 0.02
    if Wo is None:
        Wo = np.random.randn(D, D) * 0.02

    x_norm = layer_norm(x)
    Q = x_norm @ Wq
    K = x_norm @ Wk
    V = x_norm @ Wv

    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)

    out = attn @ V
    out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
    out = out @ Wo

    x = x + out

    hidden_dim = int(D * mlp_ratio)
    if W1 is None:
        W1 = np.random.randn(D, hidden_dim) * 0.02
    if W2 is None:
        W2 = np.random.randn(hidden_dim, D) * 0.02

    x_norm2 = layer_norm(x)
    mlp = x_norm2 @ W1
    mlp = 0.5 * mlp * (1 + np.tanh(np.sqrt(2 / np.pi) * (mlp + 0.044715 * mlp ** 3)))
    mlp = mlp @ W2

    x = x + mlp
    return x