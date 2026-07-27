def gae(rewards, values, gamma, lam):
    advantages = [0.0] * len(rewards)
    last_adv = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = float(last_adv)
    return advantages