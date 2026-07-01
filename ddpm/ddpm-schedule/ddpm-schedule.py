import numpy as np


def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    betas = np.linspace(beta_1, beta_T, T)
    return np.round(betas, 6)


def cosine_alpha_bar_schedule(T, s=0.008):
    ts = np.arange(1, T + 1)
    f_t = np.cos(((ts / T + s) / (1 + s)) * (np.pi / 2)) ** 2
    f_0 = np.cos((s / (1 + s)) * (np.pi / 2)) ** 2
    alpha_bars = f_t / f_0
    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)
    return np.round(alpha_bars, 6)


def alpha_bar_to_betas(alpha_bars):
    alpha_bars = np.asarray(alpha_bars, dtype=float)
    prev_alpha_bars = np.concatenate(([1.0], alpha_bars[:-1]))
    betas = 1.0 - (alpha_bars / prev_alpha_bars)
    betas = np.clip(betas, 0.0001, 0.9999)
    return np.round(betas, 6)