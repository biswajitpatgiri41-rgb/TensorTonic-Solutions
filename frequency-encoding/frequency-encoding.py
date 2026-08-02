def frequency_encoding(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = len(values)
    return [counts[value] / total for value in values]