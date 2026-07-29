def target_encoding(categories, targets):
    sums = {}
    counts = {}

    for category, target in zip(categories, targets):
        sums[category] = sums.get(category, 0) + target
        counts[category] = counts.get(category, 0) + 1

    means = {}
    for category in sums:
        means[category] = sums[category] / counts[category]

    return [float(means[category]) for category in categories]