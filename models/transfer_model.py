from abc import ABC, abstractmethod
from torch import nn
import torch


class BaseTransferModel(nn.Module, ABC):
    def __init__(self, num_classes: int, dropout_rate: float = 0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self._backbone = None
        self._backbone_layers = []
    
    @abstractmethod
    def _build_backbone(self) -> nn.Module:
        """Build and return the backbone model with modified classifier."""
        pass

    @abstractmethod
    def _collect_backbone_layers(self) -> None:
        """Fill self._backbone_layers as [layer0, layer1, ...]"""
        pass
    
    @abstractmethod
    def _get_classifier_params(self) -> list:
        """Return parameters of the classifier layer(s)."""
        pass
    
    @abstractmethod
    def _get_backbone_params(self) -> list:
        """Return parameters of the backbone (excluding classifier)."""
        pass
    
    def _apply_freezing(self, freeze_until: int = -1) -> None:
        """
        Apply layers freezing strategy.
        Args:
            freeze_until: freeze layers until this index
                  (-1=full fine tuninig; 0=feature extraction).
        """
        for p in self._get_classifier_params():
            p.requires_grad = True
        
        if freeze_until == -1:
            for p in self._get_backbone_params():
                p.requires_grad = True
            return

        for i, layer in enumerate(self._backbone_layers):
            requires_grad = i >= freeze_until
            for p in layer.parameters():
                p.requires_grad = requires_grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward pass"""
        return self._backbone(x)
    
    def get_trainable_params(self):
        """Returns iterator that yielding parameters that require gradients."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def get_param_groups(self, lr_backbone: float = 1e-4, lr_classifier: float = 1e-3) -> list[dict]:
        """
        Get parameter groups with different learning rates.
        Args:
            lr_backbone: Learning rate for backbone
            lr_classifier: Learning rate for classifier
        Returns:
            List of parameter groups for optimizer
        """
        return [
            {'params': self._get_backbone_params(), 'lr': lr_backbone},
            {'params': self._get_classifier_params(), 'lr': lr_classifier}
        ]
    
    def print_model_info(self) -> None:
        """Print information about the model architecture and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model: {self.__class__.__name__}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params/total_params:.2f}%)")