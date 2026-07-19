import numpy as np

def auc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if fpr.shape != tpr.shape or fpr.size < 2:
        raise ValueError("fpr and tpr must have the same length and contain at least 2 points")

    return float(np.trapezoid(tpr, fpr))