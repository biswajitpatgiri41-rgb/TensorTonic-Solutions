def discount_returns(rewards, gamma):
    n = len(rewards)
    G = [0.0] * n
    G[-1] = float(rewards[-1])
    for i in range(n - 2, -1, -1):
        G[i] = float(rewards[i]) + gamma * G[i + 1]
    return G