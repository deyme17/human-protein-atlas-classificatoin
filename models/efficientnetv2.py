import torch
from torch import nn
from torchvision.models import (
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights
)
from .transfer_model import BaseTransferModel
from config import DataConfig


class TransferEfficientNetV2(BaseTransferModel):
    """
    Tranfer learning model with EfficientNetV2_s/m backbone.
    """
    def __init__(self, num_classes: int = DataConfig.n_classes, freeze_until: int = -1, 
                 dropout_rate: float = 0.3, small: bool = True) -> None:
        super().__init__(num_classes, dropout_rate)
        self.in_ch = DataConfig.input_ch
        self.s = small
        self._backbone = self._build_backbone()
        self._collect_backbone_layers()
        self._apply_freezing(freeze_until)
    
    def _build_backbone(self) -> nn.Module:
        """Build and return the backbone model with modified classifier."""
        model = (efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1) if self.s else 
                 efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1))

        # replace input channels 3->4
        orig_conv1 = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=self.in_ch, # (R, G, B, Y)
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=False
        )
        # copy orig weights for rgb & avg them for new channels
        with torch.no_grad():
            model.features[0][0].weight[:, :3, :, :] = orig_conv1.weight
            rgb_weight_mean = orig_conv1.weight.mean(dim=1, keepdim=True)
            for c in range(3, self.in_ch):
                model.features[0][0].weight[:, c:c+1, :, :] = rgb_weight_mean
        
        # replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate, inplace=True),
            nn.Linear(in_features, self.num_classes)
        )
        return model
    
    def _get_classifier_params(self) -> list:
        """Return parameters of the classifier layer(s)."""
        return list(self._backbone.classifier.parameters())
    
    def _get_backbone_params(self) -> list:
        """Return parameters of the backbone (excluding classifier)."""
        return [param for name, param in self._backbone.named_parameters() if "classifier" not in name]
    
    def _collect_backbone_layers(self) -> None:
        """Return parameters of the backbone (excluding classifier)."""
        self._backbone_layers = list(self._backbone.features.children())
        self._backbone_layers.append(self._backbone.avgpool)

    def name(self) -> str:
        """Return the name of the model."""
        return "efficientnet_v2"