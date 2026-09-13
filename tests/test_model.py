import pytest


def test_mobilenet_head_is_replaced_and_backbone_frozen() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from src.training.model import build_mobilenetv3_large

    model = build_mobilenetv3_large(num_classes=13, freeze_backbone=True)
    assert model.classifier[-1].out_features == 13
    assert not any(parameter.requires_grad for parameter in model.features.parameters())
