def percent_change(series):
    result = []
    for i in range(1, len(series)):
        if series[i - 1] == 0:
            result.append(0.0)
        else:
            result.append((series[i] - series[i - 1]) / series[i - 1])
    return result