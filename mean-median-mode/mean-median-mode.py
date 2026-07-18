import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.array(x)

    mean = float(np.mean(x))
    median = float(np.median(x))

    counts = Counter(x)
    max_freq = max(counts.values())
    mode = float(min(k for k, v in counts.items() if v == max_freq))

    return (mean, median, mode)