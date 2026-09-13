"""Unit tests for FastAPI backend endpoints."""
from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["num_classes"] == 13
    assert data["rag_ready"] is True


def test_weather_endpoint():
    response = client.get("/weather?city=Pune")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "temperature_c" in data
    assert "humidity_percentage" in data


def test_predict_endpoint_valid_image():
    test_dir = Path("data/test/Tomato_healthy")
    sample_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images found")

    with open(sample_files[0], "rb") as f:
        response = client.post(
            "/predict",
            files={"image": ("test_leaf.jpg", f, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop"] is not None
    assert data["disease"] is not None


def test_predict_rejects_dark_image():
    # Synthetic black image
    dark = Image.new("RGB", (150, 150), (5, 5, 5))
    buf = io.BytesIO()
    dark.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"image": ("dark.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "quality_check_failed"


def test_tts_endpoint():
    response = client.post(
        "/tts",
        data={"text": "Hello farmer, this is a health test.", "language": "english"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 1000
