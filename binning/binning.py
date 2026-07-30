def binning(values, num_bins):
    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        return [0] * len(values)

    bin_width = (max_val - min_val) / num_bins
    result = []

    for value in values:
        bin_idx = int((value - min_val) / bin_width)
        result.append(min(bin_idx, num_bins - 1))

    return result