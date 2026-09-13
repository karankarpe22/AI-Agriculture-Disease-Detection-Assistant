"""Unit tests for Image Quality Checker module."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.preprocessing.quality_check import ImageQualityChecker, check_image_quality


def test_valid_image_passes_quality_check():
    # Create synthetic high-contrast textured image
    arr = np.random.randint(50, 200, size=(224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    result = check_image_quality(img)
    assert result["is_valid"] is True
    assert len(result["issues"]) == 0
    assert result["metrics"]["width"] == 224
    assert result["metrics"]["height"] == 224


def test_too_small_resolution_fails():
    checker = ImageQualityChecker(min_width=100, min_height=100)
    small_img = Image.new("RGB", (50, 50), color=(100, 150, 100))

    result = checker.assess_image(small_img)
    assert result["is_valid"] is False
    assert "low_resolution" in result["issues"]


def test_extremely_dark_image_fails():
    checker = ImageQualityChecker()
    dark_img = Image.new("RGB", (150, 150), color=(10, 10, 10))

    result = checker.assess_image(dark_img)
    assert result["is_valid"] is False
    assert "too_dark" in result["issues"]


def test_extremely_bright_image_fails():
    checker = ImageQualityChecker()
    bright_img = Image.new("RGB", (150, 150), color=(250, 250, 250))

    result = checker.assess_image(bright_img)
    assert result["is_valid"] is False
    assert "too_bright" in result["issues"]


def test_excessively_blurry_image_fails():
    checker = ImageQualityChecker(blur_threshold=35.0)
    # Uniform flat image has zero Laplacian variance
    flat_img = Image.new("RGB", (150, 150), color=(128, 128, 128))

    result = checker.assess_image(flat_img)
    assert result["is_valid"] is False
    assert "excessive_blur" in result["issues"]
