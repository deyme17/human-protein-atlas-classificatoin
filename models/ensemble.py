from torch import nn
import torch


class Ensemble(nn.Module):
    def __init__(self, models: list[nn.Module], weights: list[float]|None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
            self.W = w / w.sum()
        else:
            self.W = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([model(X) for model in self.models])

        if self.W is not None:
            w = self.W.to(X.device)
            # reshape for broadcasting: (n_models, 1, 1, ...)
            for _ in range(outputs.dim() - 1):
                w = w.unsqueeze(-1)
            return (outputs * w).sum(dim=0)

        return outputs.mean(dim=0)