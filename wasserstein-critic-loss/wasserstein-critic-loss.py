import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    return float(np.mean(fake_scores) - np.mean(real_scores))