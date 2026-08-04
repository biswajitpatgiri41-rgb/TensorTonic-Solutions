def moving_median(values, window_size):
    result = []
    for i in range(len(values) - window_size + 1):
        window = sorted(values[i:i + window_size])
        if window_size % 2 == 1:
            median = float(window[window_size // 2])
        else:
            median = (window[window_size // 2 - 1] + window[window_size // 2]) / 2.0
        result.append(median)
    return result