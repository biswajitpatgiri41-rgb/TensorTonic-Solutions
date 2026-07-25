def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    if current_step < warmup_steps:
        if warmup_steps == 0:
            return base_lr
        return base_lr * (current_step / warmup_steps)
    return base_lr * ((total_steps - current_step) / (total_steps - warmup_steps))