import numpy as np

def streaming_minmax_init(D):
    return {
        "min": np.full(D, np.inf, dtype=float),
        "max": np.full(D, -np.inf, dtype=float),
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    X_batch = np.asarray(X_batch, dtype=float)
    batch_min = X_batch.min(axis=0)
    batch_max = X_batch.max(axis=0)
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)
    denom = np.maximum(state["max"] - state["min"], eps)
    return (X_batch - state["min"]) / denom