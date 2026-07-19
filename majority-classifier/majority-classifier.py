import numpy as np

def majority_classifier(y_train, X_test):
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]
    return np.full(len(X_test), majority_class, dtype=int)