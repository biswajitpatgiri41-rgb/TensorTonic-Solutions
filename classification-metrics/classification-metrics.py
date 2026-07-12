import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    accuracy = float(np.mean(y_true == y_pred))

    labels = np.unique(np.concatenate((y_true, y_pred)))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)

    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(float)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)

    if average == "micro":
        TP = tp.sum()
        FP = fp.sum()
        FN = fn.sum()
        p = TP / (TP + FP) if TP + FP > 0 else 0.0
        r = TP / (TP + FN) if TP + FN > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    elif average == "macro":
        p = float(np.mean(precision))
        r = float(np.mean(recall))
        f = float(np.mean(f1))
    elif average == "weighted":
        total = support.sum()
        if total == 0:
            p = r = f = 0.0
        else:
            w = support / total
            p = float(np.sum(precision * w))
            r = float(np.sum(recall * w))
            f = float(np.sum(f1 * w))
    elif average == "binary":
        if pos_label not in label_to_idx:
            p = r = f = 0.0
        else:
            i = label_to_idx[pos_label]
            p = float(precision[i])
            r = float(recall[i])
            f = float(f1[i])
    else:
        raise ValueError("average must be 'micro', 'macro', 'weighted', or 'binary'")

    return {
        "accuracy": accuracy,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f)
    }