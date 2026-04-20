import torch

class PerImageNormalize:
    """Normalize each channel by its own mean and std."""
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: (C, H, W)
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std  = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + self.eps)