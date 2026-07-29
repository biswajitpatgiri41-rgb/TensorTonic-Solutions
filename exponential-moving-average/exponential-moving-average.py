def exponential_moving_average(values, alpha):
    ema = [float(values[0])]
    for i in range(1, len(values)):
        ema.append(alpha * values[i] + (1 - alpha) * ema[-1])
    return ema