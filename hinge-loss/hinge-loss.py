import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not np.all(np.isin(y_true, (-1, 1))):
        raise ValueError("y_true must contain only -1 and +1")

    loss = np.maximum(0.0, margin - y_true * y_score)

    if reduction == "mean":
        return float(loss.mean())
    if reduction == "sum":
        return float(loss.sum())
    raise ValueError("reduction must be 'mean' or 'sum'")