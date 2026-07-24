def detect_drift(reference_counts, production_counts, threshold):
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)
    ref = [x / ref_total for x in reference_counts]
    prod = [x / prod_total for x in production_counts]
    score = 0.5 * sum(abs(p - q) for p, q in zip(ref, prod))
    return {
        "score": score,
        "drift_detected": score > threshold
    }