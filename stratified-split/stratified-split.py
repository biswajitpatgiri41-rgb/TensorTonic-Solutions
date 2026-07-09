import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.asarray(X)
    y = np.asarray(y)

    train_idx = []
    test_idx = []

    classes, counts = np.unique(y, return_counts=True)

    for cls, count in zip(classes, counts):
        idx = np.where(y == cls)[0]

        if rng is None:
            np.random.shuffle(idx)
        else:
            rng.shuffle(idx)

        n_test = int(round(count * test_size))
        if count > 1:
            n_test = min(n_test, count - 1)
        else:
            n_test = 0

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.sort(np.array(train_idx))
    test_idx = np.sort(np.array(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]