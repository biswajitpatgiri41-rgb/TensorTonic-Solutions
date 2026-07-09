import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    order = np.lexsort((np.arange(len(y_score)), -y_score))
    y_true = y_true[order]
    y_score = y_score[order]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    distinct = np.where(np.diff(y_score) != 0)[0]
    idx = np.r_[distinct, len(y_score) - 1]

    tpr = tp[idx] / P if P > 0 else np.zeros(len(idx), dtype=float)
    fpr = fp[idx] / N if N > 0 else np.zeros(len(idx), dtype=float)
    thresholds = y_score[idx]

    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds