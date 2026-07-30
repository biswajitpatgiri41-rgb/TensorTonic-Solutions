def interaction_features(X):
    result = []
    for row in X:
        new_row = list(row)
        d = len(row)
        for i in range(d):
            for j in range(i + 1, d):
                new_row.append(row[i] * row[j])
        result.append(new_row)
    return result