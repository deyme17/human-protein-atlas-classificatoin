import torch.optim.lr_scheduler as lr_scheduler
from config import TrainConfig


def step(optimizer, last_epoch=-1, step_size=80, gamma=0.1, **_):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
  return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def multi_step(optimizer, last_epoch=-1, milestones=None, gamma=0.1, **_):
    if milestones is None:
        milestones = [500, 5000]
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)


def exponential(optimizer, last_epoch=-1, gamma=0.995, **_):
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)


def reduce_lr_on_plateau(optimizer, last_epoch=-1, mode="min", factor=0.1, patience=10,
                         threshold=0.001, threshold_mode="rel", cooldown=0, min_lr=0, **_):
    # last_epoch is unused — ReduceLROnPlateau doesn't accept it
    return lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience,
        threshold=threshold, threshold_mode=threshold_mode,
        cooldown=cooldown, min_lr=min_lr,
    )


def cosine(optimizer, last_epoch=-1, T_max=50, eta_min=1e-6, **_):
    print(f"cosine annealing — T_max: {T_max}, eta_min: {eta_min}, last_epoch: {last_epoch}")
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)


_SCHEDULERS = {
    "step": step,
    "multi_step": multi_step,
    "exponential": exponential,
    "reduce_lr_on_plateau": reduce_lr_on_plateau,
    "cosine": cosine,
}


def get_scheduler(optimizer, last_epoch: int = -1):
    cfg = TrainConfig.lr_scheduler
    name = cfg["name"].lower()
    print(f"LR Scheduler: {name}")
    params = cfg.get("parameters", {})
    if name not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(_SCHEDULERS)}")
    return _SCHEDULERS[name](optimizer, last_epoch=last_epoch, **params)