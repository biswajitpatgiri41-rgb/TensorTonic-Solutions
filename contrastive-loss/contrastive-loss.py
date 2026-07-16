import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    Compute contrastive loss for Siamese pairs.

    Args:
        a, b: Arrays of shape (N, D) or (D,). Inputs are broadcast to (N, D).
        y: Array of shape (N,) with values in {0,1}.
           1 = similar (positive), 0 = dissimilar (negative).
        margin: Positive margin for negative pairs.
        reduction: "mean" or "sum".

    Returns:
        float: Reduced contrastive loss.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if margin <= 0:
        raise ValueError("margin must be > 0")

    # Ensure embeddings are 2D
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError("a must have shape (D,) or (N,D)")

    if b.ndim == 1:
        b = b[None, :]
    elif b.ndim != 2:
        raise ValueError("b must have shape (D,) or (N,D)")

    # Broadcast embeddings
    try:
        a, b = np.broadcast_arrays(a, b)
    except ValueError:
        raise ValueError("a and b cannot be broadcast to the same shape")

    n = a.shape[0]

    # Validate y
    y = np.asarray(y).reshape(-1)
    if y.size == 1:
        y = np.full(n, y.item(), dtype=np.float64)
    elif y.size != n:
        raise ValueError("y must have length N or be a scalar")

    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distances
    d = np.linalg.norm(a - b, axis=1)

    # Contrastive loss
    loss = y * (d ** 2) + (1.0 - y) * (np.maximum(0.0, margin - d) ** 2)

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")