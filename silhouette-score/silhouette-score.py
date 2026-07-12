import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    unique_labels = np.unique(labels)
    n = len(labels)

    a = np.zeros(n, dtype=float)
    b = np.full(n, np.inf, dtype=float)

    for c in unique_labels:
        in_cluster = labels == c
        m = np.sum(in_cluster)

        if m > 1:
            a[in_cluster] = (D[np.ix_(in_cluster, in_cluster)].sum(axis=1)) / (m - 1)
        else:
            a[in_cluster] = 0.0

        for oc in unique_labels:
            if oc == c:
                continue
            other_cluster = labels == oc
            mean_dist = D[np.ix_(in_cluster, other_cluster)].mean(axis=1)
            b[in_cluster] = np.minimum(b[in_cluster], mean_dist)

    s = np.where(np.maximum(a, b) > 0, (b - a) / np.maximum(a, b), 0.0)
    return float(np.mean(s))