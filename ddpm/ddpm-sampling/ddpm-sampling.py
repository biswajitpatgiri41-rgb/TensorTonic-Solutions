import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Perform DDPM reverse sampling (Ho et al., 2020, Algorithm 2).

    Parameters
    ----------
    x_T : array_like
        Initial noisy sample x_T.
    betas : array_like
        1-D array of betas (length T).
    epsilon_preds : sequence
        Noise predictions in the ORDER THEY ARE USED:
        epsilon_preds[0] is used at t = T, epsilon_preds[1] at t = T-1, ...
    z_values : sequence
        Gaussian noise samples, also in order of use:
        z_values[0] at t = T, z_values[1] at t = T-1, ...
        (length T-1, since no noise is added at t = 1)

    Returns
    -------
    np.ndarray
        Final denoised sample x_0.
    """

    x = np.asarray(x_T, dtype=float)
    betas = np.asarray(betas, dtype=float)

    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)

    T = len(betas)

    for t in range(T, 0, -1):

        alpha = alphas[t - 1]          # schedule value for step t
        alpha_bar = alpha_bars[t - 1]  # cumulative product up to t

        # epsilon_preds are supplied in order of use (t = T first)
        eps = np.asarray(epsilon_preds[T - t], dtype=float)

        coef = (1.0 - alpha) / np.sqrt(1.0 - alpha_bar)
        x = (x - coef * eps) / np.sqrt(alpha)

        if t > 1:
            # z_values also in order of use (t = T first); z = 0 at t = 1
            z = np.asarray(z_values[T - t], dtype=float)
            x = x + np.sqrt(betas[t - 1]) * z

    return x