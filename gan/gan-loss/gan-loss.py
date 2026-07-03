import numpy as np

def discriminator_loss(real_probs, fake_probs):
    real_probs = np.clip(real_probs, 1e-8, 1 - 1e-8)
    fake_probs = np.clip(fake_probs, 1e-8, 1 - 1e-8)
    return round(float(-np.mean(np.log(real_probs) + np.log(1 - fake_probs))), 4)

def generator_loss(fake_probs):
    fake_probs = np.clip(fake_probs, 1e-8, 1 - 1e-8)
    return round(float(-np.mean(np.log(fake_probs))), 4)