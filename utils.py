import math

def get_cosine_schedule_with_warmup_lr_lambda(current_step, *, min_lr, num_warmup_steps, num_training_steps, num_cycles=0.5):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(max(min_lr, 0.0), 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))