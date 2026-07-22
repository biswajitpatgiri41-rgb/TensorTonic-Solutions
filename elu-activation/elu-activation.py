import math

def elu(x, alpha):
    return [float(v) if v > 0 else float(alpha * (math.exp(v) - 1)) for v in x]