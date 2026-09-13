"""End-to-End integration test for the AI Agriculture Assistant pipeline."""
from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_full_analyze_pipeline_english():
    test_dir = Path("data/test/Potato___Early_blight")
    sample_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images found")

    with open(sample_files[0], "rb") as f:
        response = client.post(
            "/analyze",
            files={"image": ("potato_leaf.jpg", f, "image/jpeg")},
            data={
                "question": "What fungicide should I use to stop early blight?",
                "location": "Nashik",
                "language": "english",
                "include_gradcam": "true",
            },
        )

    assert response.status_code == 200
    data = response.json()

    # 1. Verification of Image Quality Check
    assert data["success"] is True
    assert data["quality"]["is_valid"] is True

    # 2. Verification of MobileNetV3 Disease Prediction
    assert data["crop"] == "Potato"
    assert "blight" in data["disease"].lower()
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_predictions"]) >= 1

    # 3. Verification of Weather Context
    assert data["weather"]["success"] is True
    assert "temperature_c" in data["weather"]

    # 4. Verification of RAG Retrieval & Sources
    assert len(data["evidence"]) > 0
    assert len(data["sources"]) > 0

    # 5. Verification of Guidance
    assert "explanation" in data["guidance"]
    assert "management_guidance" in data["guidance"]
    assert "prevention" in data["guidance"]
    assert data["guidance"]["language"] == "english"

    # 6. Verification of Grad-CAM Visual Heatmap
    assert data["gradcam_base64"] is not None
    assert len(data["gradcam_base64"]) > 100


def test_full_analyze_pipeline_marathi():
    test_dir = Path("data/test/Tomato_healthy")
    sample_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images found")

    with open(sample_files[0], "rb") as f:
        response = client.post(
            "/analyze",
            files={"image": ("tomato_healthy.jpg", f, "image/jpeg")},
            data={
                "question": "पिकाची निगा कशी राखावी?",
                "location": "Pune",
                "language": "marathi",
                "include_gradcam": "false",
            },
        )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["crop"] == "Tomato"
    assert data["guidance"]["language"] == "marathi"
    assert len(data["guidance"]["explanation"]) > 10
