import numpy as np

def get_alpha_bar(betas):
    alpha_bar = np.cumprod(1 - np.array(betas))
    return [round(float(a), 6) for a in alpha_bar]

def forward_diffusion(x_0, t, betas, epsilon):
    x_0 = np.array(x_0)
    epsilon = np.array(epsilon)
    alpha_bar = np.cumprod(1 - np.array(betas))
    alpha_bar_t = alpha_bar[t - 1]
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
    x_t = np.round(x_t, 4)
    return x_t.tolist()