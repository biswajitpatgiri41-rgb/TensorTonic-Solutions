from math import ceil

def evaluate_shadow(production_log, shadow_log, criteria):
    n = len(production_log)

    production_accuracy = sum(
        1 for x in production_log if x["prediction"] == x["actual"]
    ) / n

    shadow_accuracy = sum(
        1 for x in shadow_log if x["prediction"] == x["actual"]
    ) / n

    accuracy_gain = shadow_accuracy - production_accuracy

    latencies = sorted(x["latency_ms"] for x in shadow_log)
    p95_index = ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[p95_index]

    agreement_rate = sum(
        1
        for p, s in zip(production_log, shadow_log)
        if p["prediction"] == s["prediction"]
    ) / n

    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate,
        },
    }