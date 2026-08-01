def weighted_moving_average(values, weights):
    k = len(weights)
    w_sum = sum(weights)
    result = []
    for i in range(len(values) - k + 1):
        weighted_sum = sum(weights[j] * values[i + j] for j in range(k))
        result.append(weighted_sum / w_sum)
    return result