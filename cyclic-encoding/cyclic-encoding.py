import math

def cyclic_encoding(values, period):
    result = []
    for value in values:
        angle = 2 * math.pi * value / period
        result.append([math.sin(angle), math.cos(angle)])
    return result