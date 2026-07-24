def expected_calibration_error(y_true, y_pred, n_bins):
    n = len(y_true)
    ece = 0.0

    for b in range(n_bins):
        indices = []
        for i, p in enumerate(y_pred):
            bin_idx = min(int(p * n_bins), n_bins - 1)
            if bin_idx == b:
                indices.append(i)

        if not indices:
            continue

        bin_size = len(indices)
        acc = sum(y_true[i] for i in indices) / bin_size
        conf = sum(y_pred[i] for i in indices) / bin_size
        ece += (bin_size / n) * abs(acc - conf)

    return ece