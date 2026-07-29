def simple_moving_average(values, window_size):
    result = []
    for i in range(len(values) - window_size + 1):
        result.append(sum(values[i:i + window_size]) / window_size)
    return result