import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    if system_type == "classification":
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        n = len(y_true)
        accuracy = (tp + tn) / n if n else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall),
        ]
    elif system_type == "regression":
        n = len(y_true)
        errors = [yt - yp for yt, yp in zip(y_true, y_pred)]
        mae = sum(abs(e) for e in errors) / n if n else 0.0
        rmse = math.sqrt(sum(e * e for e in errors) / n) if n else 0.0
        metrics = [
            ("mae", mae),
            ("rmse", rmse),
        ]
    elif system_type == "ranking":
        paired = sorted(zip(y_pred, y_true), key=lambda x: x[0], reverse=True)
        top3 = paired[:3]
        relevant_top3 = sum(label for _, label in top3)
        total_relevant = sum(y_true)
        precision_at_3 = relevant_top3 / 3
        recall_at_3 = relevant_top3 / total_relevant if total_relevant else 0.0
        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]
    else:
        raise ValueError("Invalid system type")
    return sorted(metrics, key=lambda x: x[0])