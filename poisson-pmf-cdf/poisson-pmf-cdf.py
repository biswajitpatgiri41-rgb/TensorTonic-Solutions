import numpy as np

def poisson_pmf_cdf(lam, k):
    def log_factorial(n):
        if n <= 1:
            return 0.0
        return np.sum(np.log(np.arange(1, n + 1)))

    pmf = np.exp(-lam + k * np.log(lam) - log_factorial(k))

    cdf = 0.0
    for i in range(k + 1):
        cdf += np.exp(-lam + i * np.log(lam) - log_factorial(i))

    return float(pmf), float(cdf)