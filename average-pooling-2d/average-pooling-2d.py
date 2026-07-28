def average_pooling_2d(X, pool_size):
    h = len(X)
    w = len(X[0])
    out_h = h // pool_size
    out_w = w // pool_size
    output = []
    for i in range(out_h):
        row = []
        for j in range(out_w):
            total = 0
            for a in range(pool_size):
                for b in range(pool_size):
                    total += X[i * pool_size + a][j * pool_size + b]
            row.append(total / (pool_size * pool_size))
        output.append(row)
    return output