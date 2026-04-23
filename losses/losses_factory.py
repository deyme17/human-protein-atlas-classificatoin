import torch
import torch.nn as nn
from config import TrainConfig, DataConfig
from .focal_loss import FocalLoss
from .asymmetric_loss import AsymmetricLoss


def focal(alphas=None, gamma=2.0, reduction="mean", device=TrainConfig.device, **_):
    if alphas is None:
        alphas = torch.ones(DataConfig.n_classes)
    return FocalLoss(alphas=alphas.to(device), gamma=gamma, reduction=reduction)


def asymmetric(gamma_neg=4, gamma_pos=1, clip=0.05, **_):
    return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)


def bce(pos_weight=None, reduction="mean", **_):
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)


def cross_entropy(label_smoothing=0.0, weight=None, reduction="mean", **_):
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction=reduction)


_LOSSES = {
    "focal": focal,
    "bce": bce,
    "cross_entropy": cross_entropy,
    "asymmetric": asymmetric
}


def get_loss() -> nn.Module:
    cfg = TrainConfig.loss
    name = cfg["name"].lower()
    print(f"Loss function: {name}")
    params = cfg.get("parameters", {})
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSSES)}")
    return _LOSSES[name](**params)