def autocorrelation(series, max_lag):
    n = len(series)
    mean = sum(series) / n
    variance = sum((x - mean) ** 2 for x in series)

    if variance == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for lag in range(max_lag + 1):
        covariance = 0.0
        for t in range(n - lag):
            covariance += (series[t] - mean) * (series[t + lag] - mean)
        result.append(covariance / variance)

    return result