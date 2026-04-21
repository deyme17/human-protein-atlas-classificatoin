import torch.optim as optim
from config import TrainConfig


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)


def adamw(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)


_OPTIMIZERS = {
    "adam": adam,
    "adamw": adamw,
    "sgd": sgd,
}


def get_optimizer(param_groups) -> optim.Optimizer:
    cfg = TrainConfig.optimizer
    name = cfg["name"].lower()
    params = cfg.get("parameters", {})
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](param_groups, **params)