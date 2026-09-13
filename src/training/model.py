"""Pretrained MobileNetV3-Large classifier construction and layer manipulation."""
from __future__ import annotations

from typing import Any, cast
import torch.nn as nn
from torchvision.models import MobileNetV3, MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_mobilenetv3_large(num_classes: int, freeze_backbone: bool = True) -> Any:
    """Build ImageNet-pretrained MobileNetV3-Large with a new disease head."""
    model: Any = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    if freeze_backbone:
        features: Any = getattr(model, "features", None)
        if features is not None:
            for parameter in features.parameters():
                parameter.requires_grad = False

    classifier: Any = getattr(model, "classifier", None)
    if classifier is not None:
        last_layer: Any = classifier[-1]
        classifier_in_features: int = getattr(last_layer, "in_features", 1024)
        classifier[-1] = nn.Linear(classifier_in_features, num_classes)
    return model


def unfreeze_upper_layers(model: Any, num_blocks: int = 3) -> Any:
    """Unfreeze the top InvertedResidual blocks of the backbone for fine-tuning."""
    # Ensure classifier requires grad
    classifier: Any = getattr(model, "classifier", None)
    if classifier is not None:
        for parameter in classifier.parameters():
            parameter.requires_grad = True

    # Unfreeze the last num_blocks features
    features: Any = getattr(model, "features", None)
    if features is not None:
        feature_blocks = list(features)
        for block in feature_blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    return model
