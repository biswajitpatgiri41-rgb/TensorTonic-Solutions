import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    if rng is None:
        if np.random.random() < epsilon:
            return int(np.random.randint(len(q_values)))
        return int(np.argmax(q_values))
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))