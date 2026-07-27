def policy_gradient_loss(log_probs, rewards, gamma):
    T = len(rewards)
    returns = [0.0] * T
    G = 0.0
    for i in range(T - 1, -1, -1):
        G = rewards[i] + gamma * G
        returns[i] = G
    mean_return = sum(returns) / T
    loss = -sum(lp * (ret - mean_return) for lp, ret in zip(log_probs, returns)) / T
    return loss