import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    returns_sum = np.zeros(n_states, dtype=float)
    returns_count = np.zeros(n_states, dtype=int)

    for episode in episodes:
        G = 0.0
        returns = [0.0] * len(episode)
        for t in range(len(episode) - 1, -1, -1):
            _, reward = episode[t]
            G = reward + gamma * G
            returns[t] = G

        visited = set()
        for t, (state, _) in enumerate(episode):
            if state not in visited:
                visited.add(state)
                returns_sum[state] += returns[t]
                returns_count[state] += 1

    V = np.zeros(n_states, dtype=float)
    mask = returns_count > 0
    V[mask] = returns_sum[mask] / returns_count[mask]
    return V