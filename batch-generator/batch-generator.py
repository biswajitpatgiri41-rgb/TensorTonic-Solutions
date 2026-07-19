import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)
    indices = np.arange(n)

    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    end = n if not drop_last else (n // batch_size) * batch_size

    for i in range(0, end, batch_size):
        yield X_shuffled[i:i + batch_size], y_shuffled[i:i + batch_size]