def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if step >= total_steps:
        return float(final_lr)

    if warmup_steps > 0 and step <= warmup_steps:
        return float(initial_lr * (step / warmup_steps))

    if total_steps == warmup_steps:
        return float(final_lr)

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = initial_lr + (final_lr - initial_lr) * decay_progress
    return float(lr)