import numpy as np

def compute_advantage(states, rewards, V, gamma):
    n = len(rewards)
    returns = np.empty(n, dtype=float)
    G = 0.0
    for i in range(n - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = G
    return np.array([returns[i] - V[states[i]] for i in range(n)])