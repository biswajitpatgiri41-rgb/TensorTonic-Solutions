import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum(y_true == y_pred)
    fp = len(y_true) - tp
    fn = len(y_true) - tp

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0

    return float((2 * tp) / denom)