def maxpool_forward(X, pool_size, stride):
    H = len(X)
    W = len(X[0])
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            max_val = float('-inf')
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[i * stride + a][j * stride + b]
                    if val > max_val:
                        max_val = val
            row.append(max_val)
        output.append(row)

    return output