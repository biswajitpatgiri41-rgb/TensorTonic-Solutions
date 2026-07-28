def linear_layer_forward(X, W, b):
    n = len(X)
    d_out = len(b)
    d_in = len(W)
    Y = []
    for i in range(n):
        row = []
        for j in range(d_out):
            s = b[j]
            for k in range(d_in):
                s += X[i][k] * W[k][j]
            row.append(s)
        Y.append(row)
    return Y