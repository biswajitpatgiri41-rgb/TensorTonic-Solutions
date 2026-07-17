import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred
    abs_error = np.abs(error)
    loss = np.where(
        abs_error <= delta,
        0.5 * error ** 2,
        delta * (abs_error - 0.5 * delta)
    )
    return np.mean(loss)