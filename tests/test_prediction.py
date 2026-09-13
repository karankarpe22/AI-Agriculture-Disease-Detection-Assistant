"""Unit tests for Disease Predictor and Grad-CAM explainability."""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from src.prediction.predict import DiseasePredictor, format_class_name


def test_format_class_name():
    crop, disease = format_class_name("Tomato_Late_blight")
    assert crop == "Tomato"
    assert disease == "Late blight"

    crop2, disease2 = format_class_name("Pepper__bell___Bacterial_spot")
    assert crop2 == "Bell pepper"
    assert disease2 == "Bacterial spot"


def test_predictor_loads_and_predicts():
    predictor = DiseasePredictor()
    assert len(predictor.class_names) == 13
    assert predictor.model is not None

    # Load actual test image from data/test
    test_dir = Path("data/test/Tomato_healthy")
    sample_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_images:
        pytest.skip("No test images found in data/test/Tomato_healthy")

    img = Image.open(sample_images[0])
    result = predictor.predict(img, top_k=3)

    assert result["success"] is True
    assert result["crop"] is not None
    assert result["disease"] is not None
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3
    assert result["confidence_percentage"] == round(result["confidence"] * 100, 2)


def test_predictor_gradcam_generation():
    predictor = DiseasePredictor()
    test_dir = Path("data/test/Potato___Late_blight")
    sample_images = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_images:
        pytest.skip("No test images found in data/test/Potato___Late_blight")

    img = Image.open(sample_images[0])
    cam = predictor.generate_gradcam(img)
    assert cam is not None
    assert cam.size == img.size
