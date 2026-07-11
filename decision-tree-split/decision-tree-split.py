import numpy as np

def decision_tree_split(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)
    n_samples, n_features = X.shape

    best_gain = -1.0
    best_feature = None
    best_threshold = None

    for feature in range(n_features):
        values = np.unique(X[:, feature])
        values.sort()

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2.0

            left_mask = X[:, feature] <= threshold
            right_mask = ~left_mask

            if not left_mask.any() or not right_mask.any():
                continue

            y_left = y[left_mask]
            y_right = y[right_mask]

            split_gini = (
                len(y_left) / n_samples * gini(y_left)
                + len(y_right) / n_samples * gini(y_right)
            )

            gain = parent_gini - split_gini

            if (
                gain > best_gain
                or (
                    np.isclose(gain, best_gain)
                    and (
                        best_feature is None
                        or feature < best_feature
                        or (
                            feature == best_feature
                            and threshold < best_threshold
                        )
                    )
                )
            ):
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return [best_feature, best_threshold]