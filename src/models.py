"""
Stage 5 & Stage 6: CNN Backbone Architectures for ROP Classification.

Supports:
1. ResNet-50 (Primary baseline model).
2. EfficientNet-B0 (Lightweight comparison).
3. ConvNeXt-Tiny (Modern CNN comparison).

All models:
- Initialize with pretrained ImageNet weights.
- Feature configurable classification heads (num_classes: 2 for binary, 3 for severity).
- Provide unified get_model(name, num_classes, pretrained) factory.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """
    Factory function for CNN backbones.
    """
    model_name = model_name.lower().replace("-", "_")

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        model.backbone_name = "resnet50"

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        model.backbone_name = "efficientnet_b0"

    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        model.backbone_name = "convnext_tiny"

    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: resnet50, efficientnet_b0, convnext_tiny")

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
