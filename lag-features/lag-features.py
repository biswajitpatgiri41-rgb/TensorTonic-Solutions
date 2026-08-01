def lag_features(series, lags):
    max_lag = max(lags)
    result = []
    for t in range(max_lag, len(series)):
        result.append([series[t - lag] for lag in lags])
    return result