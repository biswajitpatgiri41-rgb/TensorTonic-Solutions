import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    recon = float(np.mean(np.sum((x - x_recon) ** 2, axis=1)))
    kl = float(np.mean(-0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)))
    total = recon + kl
    return {
        "total": float(total),
        "recon": float(recon),
        "kl": float(kl)
    }