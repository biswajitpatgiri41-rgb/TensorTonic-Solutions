import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    x_0 = np.array(x_0)
    betas = np.array(betas)
    epsilon = np.array(epsilon)
    epsilon_pred = np.array(epsilon_pred)
    t_indices = np.array(t_values) - 1

    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    alpha_bar_t = alpha_bar[t_indices]

    while alpha_bar_t.ndim < x_0.ndim:
        alpha_bar_t = np.expand_dims(alpha_bar_t, -1)

    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1.0 - alpha_bar_t) * epsilon

    loss = np.mean((epsilon - epsilon_pred) ** 2)
    return float(loss)