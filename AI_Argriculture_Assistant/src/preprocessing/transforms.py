"""MobileNetV3 image preprocessing with conservative leaf-image augmentation."""
from __future__ import annotations

from typing import Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> dict[str, Any]:
    """Return transforms for train, validation, and test/inference.

    Training changes are deliberately modest: leaf orientation and small camera
    variations can occur in practice, while vertical flips, strong colour shifts,
    and extreme rotations are omitted. Validation and test transforms are fully
    deterministic to keep metrics comparable.
    """
    try:
        from torchvision import transforms
    except ImportError as exc:  # Allows dataset tooling to work before ML deps.
        raise RuntimeError(
            "PyTorch and torchvision are required for preprocessing. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    deterministic = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    training = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    return {"train": training, "validation": deterministic, "test": deterministic}
