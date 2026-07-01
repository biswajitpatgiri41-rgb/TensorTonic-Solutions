import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    B, N, D = encoder_output.shape
    cls_token = encoder_output[:, 0, :]
    eps = 1e-6

    mean = cls_token.mean(axis=-1, keepdims=True)
    var = cls_token.var(axis=-1, keepdims=True)
    cls_norm = (cls_token - mean) / np.sqrt(var + eps)

    if W_head is None:
        W_head = np.random.randn(D, num_classes) * 0.02

    logits = cls_norm @ W_head
    return logits