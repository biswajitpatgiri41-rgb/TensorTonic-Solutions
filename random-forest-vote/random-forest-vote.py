import numpy as np

def random_forest_vote(predictions):
    predictions = np.array(predictions)
    n_samples = predictions.shape[1]
    result = []
    for i in range(n_samples):
        votes = np.bincount(predictions[:, i])
        result.append(int(np.argmax(votes)))
    return result