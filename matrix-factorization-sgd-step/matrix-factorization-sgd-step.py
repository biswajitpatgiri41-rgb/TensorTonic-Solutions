def matrix_factorization_sgd_step(U, V, r, lr, reg):
    error = r - sum(u * v for u, v in zip(U, V))
    U_new = [u + lr * (error * v - reg * u) for u, v in zip(U, V)]
    V_new = [v + lr * (error * u - reg * v) for u, v in zip(U, V)]
    return U_new, V_new