import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    real_logits = np.dot(real_data, D_W)
    fake_logits = np.dot(fake_data, D_W)

    real_probs = 1 / (1 + np.exp(-real_logits))
    fake_probs = 1 / (1 + np.exp(-fake_logits))

    real_probs = np.clip(real_probs, 1e-8, 1 - 1e-8)
    fake_probs = np.clip(fake_probs, 1e-8, 1 - 1e-8)

    d_loss = -np.mean(np.log(real_probs) + np.log(1 - fake_probs))
    g_loss = -np.mean(np.log(fake_probs))

    return {
        "d_loss": round(float(d_loss), 4),
        "g_loss": round(float(g_loss), 4)
    }