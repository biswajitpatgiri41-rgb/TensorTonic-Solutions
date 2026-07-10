def retraining_policy(daily_stats, config):
    drift_threshold = config["drift_threshold"]
    performance_threshold = config["performance_threshold"]
    max_staleness = config["max_staleness"]
    cooldown = config["cooldown"]
    retrain_cost = config["retrain_cost"]
    budget = config["budget"]

    retrain_days = []
    days_since_retrain = 0
    cooldown_counter = cooldown

    for stats in daily_stats:
        days_since_retrain += 1
        cooldown_counter += 1

        trigger = (
            stats["drift_score"] > drift_threshold or
            stats["performance"] < performance_threshold or
            days_since_retrain >= max_staleness
        )

        if trigger and cooldown_counter >= cooldown and budget >= retrain_cost:
            retrain_days.append(stats["day"])
            budget -= retrain_cost
            days_since_retrain = 0
            cooldown_counter = 0

    return retrain_days