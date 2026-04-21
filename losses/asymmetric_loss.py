from torch import nn
import torch


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, clip: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pm = torch.clamp(probs - self.clip, min=0)
        L_pos = targets * ((1 - probs)**self.gamma_pos) * torch.log(probs + self.eps)
        L_neg = (1 - targets) * (pm**self.gamma_neg) * torch.log(1 - pm + self.eps)
        loss = L_pos + L_neg
        return -loss.mean()