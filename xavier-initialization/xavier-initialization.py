import math

def xavier_initialization(W, fan_in, fan_out):
    limit = math.sqrt(6 / (fan_in + fan_out))
    return [[w * 2 * limit - limit for w in row] for row in W]