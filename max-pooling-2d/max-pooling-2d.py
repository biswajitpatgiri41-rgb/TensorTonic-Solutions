def max_pooling_2d(X, pool_size):
    h = len(X)
    w = len(X[0])
    out_h = h // pool_size
    out_w = w // pool_size
    output = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = float('-inf')
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * pool_size + a][j * pool_size + b]
                    if val > max_val:
                        max_val = val
            row.append(max_val)
        output.append(row)

    return output