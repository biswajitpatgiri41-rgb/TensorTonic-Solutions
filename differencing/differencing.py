def differencing(series, order):
    result = series[:]
    for _ in range(order):
        result = [result[i] - result[i - 1] for i in range(1, len(result))]
    return result