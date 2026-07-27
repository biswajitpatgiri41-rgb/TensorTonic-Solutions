import math

def cosine_embedding_loss(x1, x2, label, margin):
    dot = sum(a * b for a, b in zip(x1, x2))
    norm1 = math.sqrt(sum(a * a for a in x1))
    norm2 = math.sqrt(sum(b * b for b in x2))
    cos_sim = dot / (norm1 * norm2)
    if label == 1:
        return 1.0 - cos_sim
    return max(0.0, cos_sim - margin)