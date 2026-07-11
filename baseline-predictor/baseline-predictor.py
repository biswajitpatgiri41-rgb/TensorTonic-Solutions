def baseline_predict(ratings_matrix, target_pairs):
    non_zero = [r for row in ratings_matrix for r in row if r != 0]
    mu = sum(non_zero) / len(non_zero)

    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])

    user_bias = []
    for row in ratings_matrix:
        vals = [x for x in row if x != 0]
        if vals:
            user_bias.append(sum(vals) / len(vals) - mu)
        else:
            user_bias.append(0.0)

    item_bias = []
    for j in range(cols):
        vals = [ratings_matrix[i][j] for i in range(rows) if ratings_matrix[i][j] != 0]
        if vals:
            item_bias.append(sum(vals) / len(vals) - mu)
        else:
            item_bias.append(0.0)

    return [mu + user_bias[u] + item_bias[i] for u, i in target_pairs]