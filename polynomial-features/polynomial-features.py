def polynomial_features(values, degree):
    return [[x ** p for p in range(degree + 1)] for x in values]