from torch import nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, alphas: torch.Tensor, gamma: float = 2.5, reduction: str = "mean"):
        super().__init__()
        self.register_buffer('alphas', alphas)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alphas * (1 - p_t)**self.gamma
        loss = focal * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss