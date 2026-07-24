import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    result = {}
    for feature in train_dist:
        train = np.asarray(train_dist[feature], dtype=float) + eps
        serving = np.asarray(serving_dist[feature], dtype=float) + eps
        psi = np.sum((serving - train) * np.log(serving / train))
        result[feature] = {
            "psi": float(psi),
            "skewed": bool(psi >= threshold)
        }
    return result