import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    x = np.asarray(x)
    n = len(x)

    if rng is None:
        indices = np.random.randint(0, n, size=(n_bootstrap, n))
    else:
        indices = rng.integers(0, n, size=(n_bootstrap, n))

    boot_means = x[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1.0 - alpha)

    return boot_means, lower, upper