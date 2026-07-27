import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    total_loss = 0.0
    for p, y in zip(predictions, targets):
        pt = p if y == 1 else 1 - p
        total_loss += -alpha * ((1 - pt) ** gamma) * math.log(pt)
    return total_loss / len(predictions)