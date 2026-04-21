from config import TrainConfig
from models.efficientnetv2 import TransferEfficientNetV2

_MODELS = {
    "efficientnet_v2": TransferEfficientNetV2,
}

def get_model() -> TransferEfficientNetV2:
    cfg = TrainConfig.model
    name = cfg["name"].lower()
    params = cfg.get("parameters", {})
    if name not in _MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODELS)}")
    return _MODELS[name](**params)