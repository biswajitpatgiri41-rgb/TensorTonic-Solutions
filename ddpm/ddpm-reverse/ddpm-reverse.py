import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    x_t = np.array(x_t, dtype=float)
    epsilon_pred = np.array(epsilon_pred, dtype=float)
    betas = np.array(betas, dtype=float)

    alpha_t = 1.0 - betas[t - 1]
    alpha_bar_t = np.prod(1.0 - betas[:t])

    mu = (1.0 / np.sqrt(alpha_t)) * (
        x_t - (betas[t - 1] / np.sqrt(1.0 - alpha_bar_t)) * epsilon_pred
    )

    if t == 1:
        return mu

    if z is None:
        z = np.random.randn(*x_t.shape)
    else:
        z = np.array(z, dtype=float)

    sigma_t = np.sqrt(betas[t - 1])
    return mu + sigma_t * z