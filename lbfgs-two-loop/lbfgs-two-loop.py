def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    m = len(s_list)
    q = grad[:]
    rho = [1.0 / _dot(y_list[i], s_list[i]) for i in range(m)]
    alpha = [0.0] * m

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * _dot(s_list[i], q)
        q = [qj - alpha[i] * yj for qj, yj in zip(q, y_list[i])]

    sy = _dot(s_list[-1], y_list[-1])
    yy = _dot(y_list[-1], y_list[-1])
    gamma = sy / yy
    r = [gamma * qj for qj in q]

    for i in range(m):
        beta = rho[i] * _dot(y_list[i], r)
        r = [rj + sj * (alpha[i] - beta) for rj, sj in zip(r, s_list[i])]

    return [-x for x in r]