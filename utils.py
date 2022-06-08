from collections import defaultdict
import optax

def get_linear_scheduler(learning_rate: float = 3e-4, end_value: float = 0., warmup_steps: int = 10):
    return optax.linear_schedule(init_value=learning_rate, end_value=end_value, transition_steps=warmup_steps)


def get_adam_optimizer(scheduler, b1: float = 0.9, b2: float = 0.98, eps: float = 1e-8, weight_decay: float = 0.01):
    return optax.adamw(learning_rate=scheduler, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay)


def get_updates(epoch, dictlist):
    dictdata = defaultdict(list)
    for d in dictlist:
        for k, v in d.items():
            dictdata[k].append(v)

    return dict({"epoch": epoch}, **{k: (sum(v) / len(v)) for k, v in dictdata.items()})
