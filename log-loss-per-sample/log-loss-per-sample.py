import math

def log_loss(y_true, y_pred, eps=1e-15):
    return [
        -(y * math.log(min(max(p, eps), 1 - eps)) +
          (1 - y) * math.log(1 - min(max(p, eps), 1 - eps)))
        for y, p in zip(y_true, y_pred)
    ]