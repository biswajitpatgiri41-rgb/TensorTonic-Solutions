import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return float(pmf), float(cdf)