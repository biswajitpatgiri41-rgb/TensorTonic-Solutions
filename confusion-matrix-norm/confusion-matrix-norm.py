import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize="none"):
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if normalize not in ("none", "true", "pred", "all"):
        raise ValueError("normalize must be one of 'none', 'true', 'pred', or 'all'")

    if num_classes is None:
        if y_true.size == 0:
            num_classes = 0
        else:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1
    else:
        num_classes = int(num_classes)
        if num_classes < 0:
            raise ValueError("num_classes must be non-negative")

    if num_classes == 0:
        dtype = np.int64 if normalize == "none" else np.float64
        return np.zeros((0, 0), dtype=dtype)

    if y_true.size > 0:
        if (
            np.any(y_true < 0)
            or np.any(y_pred < 0)
            or np.any(y_true >= num_classes)
            or np.any(y_pred >= num_classes)
        ):
            raise ValueError("Labels must be in the range [0, num_classes - 1]")

    indices = y_true * num_classes + y_pred
    cm = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    if normalize == "none":
        return cm

    cm = cm.astype(np.float64)

    if normalize == "true":
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    if normalize == "pred":
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        return cm / denom

    total = cm.sum()
    if total == 0:
        total = 1.0
    return cm / total