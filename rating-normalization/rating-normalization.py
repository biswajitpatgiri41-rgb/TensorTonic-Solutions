def rating_normalization(matrix):
    result = []
    for row in matrix:
        rated = [x for x in row if x != 0]
        if rated:
            mean = sum(rated) / len(rated)
            result.append([float(x - mean) if x != 0 else 0.0 for x in row])
        else:
            result.append([0.0 for _ in row])
    return result