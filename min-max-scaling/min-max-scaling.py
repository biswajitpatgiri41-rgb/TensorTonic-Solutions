def min_max_scaling(data):
    rows = len(data)
    cols = len(data[0])
    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        column = [data[i][j] for i in range(rows)]
        min_val = min(column)
        max_val = max(column)
        range_val = max_val - min_val

        if range_val == 0:
            for i in range(rows):
                result[i][j] = 0.0
        else:
            for i in range(rows):
                result[i][j] = (data[i][j] - min_val) / range_val

    return result