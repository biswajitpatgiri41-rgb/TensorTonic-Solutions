def promote_model(models):
    return max(models, key=lambda m: (m["accuracy"], -m["latency"], m["timestamp"]))["name"]