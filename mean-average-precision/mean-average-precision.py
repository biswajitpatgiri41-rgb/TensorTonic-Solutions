import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true, dtype=int)
        y_score = np.asarray(y_score, dtype=float)

        order = np.argsort(-y_score, kind="stable")
        y_true = y_true[order]

        total_relevant = np.sum(y_true)
        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue

        if k is not None:
            y_true_eval = y_true[:k]
        else:
            y_true_eval = y_true

        cumulative_relevant = np.cumsum(y_true_eval)
        ranks = np.arange(1, len(y_true_eval) + 1)
        precision = cumulative_relevant / ranks

        ap = np.sum(precision * y_true_eval) / total_relevant
        ap_per_query.append(float(ap))

    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0

    return map_value, ap_per_query