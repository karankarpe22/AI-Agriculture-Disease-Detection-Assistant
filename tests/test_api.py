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
    from unittest.mock import MagicMock, patch

    mock_weather = {
        "success": True,
        "provider": "MockWeather",
        "location": "Pune, IN",
        "temperature_c": 24.5,
        "humidity_percentage": 65,
        "condition": "Partly cloudy",
        "precipitation_mm": 0.0,
        "wind_speed_kmh": 10.0,
        "risk_analysis": "Normal seasonal conditions.",
    }
    with patch("backend.main.get_weather_service") as mock_get_ws:
        mock_ws = MagicMock()
        mock_ws.get_weather.return_value = mock_weather
        mock_get_ws.return_value = mock_ws

        response = client.get("/weather?city=Pune")

    assert response.status_code == 200
    data = response.json()
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


def test_ask_endpoint_followup():
    """Test follow-up question mode with active crop and disease context."""
    from unittest.mock import MagicMock, patch

    payload = {
        "crop": "Tomato",
        "disease": "Early blight",
        "confidence": 0.92,
        "question": "Can I spray neem oil for this early blight?",
        "location": "Pune",
        "language": "english",
    }
    mock_guidance = {
        "en": {"direct_answer": "Neem oil spray provides mild suppression, but copper oxychloride is recommended for active blight."},
        "mr": {"direct_answer": "कडुनिंब तेल फवारणी प्राथमिक नियंत्रण करते."},
        "direct_answer_en": "Neem oil spray provides mild suppression, but copper oxychloride is recommended for active blight.",
        "direct_answer_mr": "कडुनिंब तेल फवारणी प्राथमिक नियंत्रण करते.",
        "direct_answer": "Neem oil spray provides mild suppression, but copper oxychloride is recommended for active blight.",
        "explanation": "Early blight is a fungal infection caused by Alternaria solani.",
        "management_guidance": "Spray Mancozeb 75% WP @ 2.5 g/L as per ICAR guidelines.",
        "prevention": "Ensure wide spacing and avoid overhead irrigation.",
        "precautions": "Wear personal protective equipment during chemical application.",
        "language": "english",
    }

    with patch("src.graph.nodes.get_gemini_service") as mock_get_srv:
        mock_srv = MagicMock()
        mock_srv.generate_guidance.return_value = mock_guidance
        mock_get_srv.return_value = mock_srv

        response = client.post("/ask", json=payload)
        # Verify graph calls Gemini guidance service exactly once
        mock_srv.generate_guidance.assert_called_once()

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop"] == "Tomato"
    assert data["disease"] == "Early blight"
    assert "direct_answer" in data
    assert len(data["direct_answer"]) > 10
    assert "guidance" in data
    assert "evidence" in data


def test_ask_endpoint_standalone():
    """Test standalone Q&A mode without image or predefined crop/disease."""
    from unittest.mock import MagicMock, patch

    payload = {
        "crop": None,
        "disease": None,
        "question": "What is the best organic manure for potato cultivation?",
        "location": "Nashik",
        "language": "english",
    }
    mock_guidance = {
        "en": {"direct_answer": "Well-decomposed Farm Yard Manure (FYM) @ 25 t/ha supplemented with vermicompost is optimal."},
        "mr": {"direct_answer": "बटाटा पिकासाठी चांगले कुजलेले शेणखत किंवा गांडूळ खत सर्वोत्तम आहे."},
        "direct_answer_en": "Well-decomposed Farm Yard Manure (FYM) @ 25 t/ha supplemented with vermicompost is optimal.",
        "direct_answer_mr": "बटाटा पिकासाठी चांगले कुजलेले शेणखत किंवा गांडूळ खत सर्वोत्तम आहे.",
        "direct_answer": "Well-decomposed Farm Yard Manure (FYM) @ 25 t/ha supplemented with vermicompost is optimal.",
        "explanation": "Organic matter enhances soil aeration and tuber enlargement.",
        "management_guidance": "Incorporate FYM 3-4 weeks prior to planting.",
        "prevention": "Ensure certified disease-free tubers.",
        "precautions": "Avoid raw or unfermented manure.",
        "language": "english",
    }

    with patch("src.graph.nodes.get_gemini_service") as mock_get_srv:
        mock_srv = MagicMock()
        mock_srv.generate_guidance.return_value = mock_guidance
        mock_get_srv.return_value = mock_srv

        response = client.post("/ask", json=payload)
        # Verify graph calls Gemini guidance service exactly once
        mock_srv.generate_guidance.assert_called_once()

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "direct_answer" in data
    assert len(data["direct_answer"]) > 10
    assert "guidance" in data


def test_tts_endpoint():
    from unittest.mock import MagicMock, patch

    mock_audio = b"ID3\x03\x00\x00\x00" + b"AUDIO_DATA" * 120
    with patch("backend.main.get_voice_service") as mock_voice_get:
        mock_srv = MagicMock()
        mock_srv.text_to_speech_bytes.return_value = (True, mock_audio, "audio/mpeg")
        mock_voice_get.return_value = mock_srv

        response = client.post(
            "/tts",
            data={"text": "Hello farmer, this is a health test.", "language": "english"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 1000


def test_bilingual_fields_in_ask_endpoint():
    from unittest.mock import MagicMock, patch

    payload = {
        "crop": "Potato",
        "disease": "Late blight",
        "confidence": 0.95,
        "question": "What care should be taken to prevent this disease",
        "location": "Pune",
        "language": "english",
    }
    mock_guidance = {
        "en": {"direct_answer": "Preventive spray of Mancozeb before disease onset is critical."},
        "mr": {"direct_answer": "रोगाचा प्रादुर्भाव होण्यापूर्वी मँकोझेबची प्रतिबंधात्मक फवारणी करणे अत्यंत महत्त्वाचे आहे."},
        "direct_answer_en": "Preventive spray of Mancozeb before disease onset is critical.",
        "direct_answer_mr": "रोगाचा प्रादुर्भाव होण्यापूर्वी मँकोझेबची प्रतिबंधात्मक फवारणी करणे अत्यंत महत्त्वाचे आहे.",
        "direct_answer": "Preventive spray of Mancozeb before disease onset is critical.",
        "explanation": "Late blight is devastating under cool and humid microclimates.",
        "management_guidance": "Spray systemic fungicides if active lesions appear.",
        "prevention": "Destroy cull piles and practice crop rotation.",
        "precautions": "Follow PHI waiting periods.",
        "language": "english",
    }

    with patch("src.graph.nodes.get_gemini_service") as mock_get_srv:
        mock_srv = MagicMock()
        mock_srv.generate_guidance.return_value = mock_guidance
        mock_get_srv.return_value = mock_srv

        response = client.post("/ask", json=payload)
        mock_srv.generate_guidance.assert_called_once()

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop_en"] == "Potato"
    assert data["crop_mr"] == "बटाटा"
    assert "Late blight" in data["disease_en"]
    assert "करपा" in data["disease_mr"]
    assert "direct_answer_en" in data
    assert "direct_answer_mr" in data
    assert len(data["direct_answer_mr"]) > 10
    assert "location_mr" in data["weather"]
    assert "पुणे" in data["weather"]["location_mr"]


def test_translate_endpoint():
    payload = {
        "text": "What care should be taken to prevent this disease",
        "target_lang": "marathi",
    }
    response = client.post("/translate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "काळजी" in data["translated_text"] or len(data["translated_text"]) > 5


def test_soil_endpoint():
    """Verify GET /soil endpoint returns validated synthetic record with interpretation."""
    response = client.get("/soil?city=Pune&crop=Tomato")
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Pune"
    assert data["crop_type"] == "Tomato"
    assert data["is_synthetic"] is True
    assert data["data_type"] == "synthetic"
    assert "units" in data
    assert "interpretation" in data
    assert 6.0 <= data["soil_ph"] <= 8.5
    assert "disclaimer" in data


def test_soil_endpoint_fallback():
    """Verify GET /soil endpoint falls back gracefully for unknown location."""
    response = client.get("/soil?city=NonExistentTown123")
    assert response.status_code == 200
    data = response.json()
    assert data["is_synthetic"] is True
    assert data["is_fallback"] is True
    assert "interpretation" in data


def test_health_reports_soil_ready():
    """Verify /health endpoint reports soil_ready=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "soil_ready" in data
    assert data["soil_ready"] is True


def test_analyze_endpoint_valid_image_executes_graph():
    """Verify POST /analyze executes full LangGraph workflow and returns backward-compatible payload."""
    from unittest.mock import patch
    test_dir = Path("data/test/Tomato_healthy")
    sample_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images found")

    mock_guidance = {
        "en": {"direct_answer": "Apply standard preventive measures."},
        "mr": {"direct_answer": "मानक प्रतिबंधात्मक उपाययोजना करा."},
        "direct_answer_en": "Apply standard preventive measures.",
        "direct_answer_mr": "मानक प्रतिबंधात्मक उपाययोजना करा.",
    }

    with patch("src.graph.nodes.get_gemini_service") as mock_ggs:
        mock_ggs.return_value.generate_guidance.return_value = mock_guidance
        with open(sample_files[0], "rb") as f:
            response = client.post(
                "/analyze",
                files={"image": ("test_leaf.jpg", f, "image/jpeg")},
                data={"location": "Pune", "language": "english", "question": "How to care?"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["crop"] is not None
    assert data["disease"] is not None
    assert data["confidence"] > 0.0
    assert "weather" in data
    assert "soil" in data
    assert "evidence" in data
    assert "guidance" in data
    assert "direct_answer" in data
    assert data["soil"]["is_synthetic"] is True


def test_analyze_endpoint_quality_failure_executes_graph():
    """Verify POST /analyze routes directly to END on quality check failure without fabricated diagnosis."""
    dark = Image.new("RGB", (150, 150), (5, 5, 5))
    buf = io.BytesIO()
    dark.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/analyze",
        files={"image": ("dark.jpg", buf.getvalue(), "image/jpeg")},
        data={"location": "Pune", "language": "english"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "quality_check_failed"
    assert data["crop"] is None
    assert data["disease"] is None
    assert data["confidence"] == 0.0
    assert data["warning"] is not None


def test_analyze_endpoint_gradcam_toggle_executes_graph():
    """Verify POST /analyze generates raw base64 Grad-CAM when requested."""
    from unittest.mock import patch
    test_dir = Path("data/test/Tomato_healthy")
    sample_files = list(test_dir.glob("*.JPG")) + list(test_dir.glob("*.jpg"))
    if not sample_files:
        pytest.skip("No sample images found")

    mock_guidance = {
        "en": {"direct_answer": "Preventive spray guidance."},
        "direct_answer_en": "Preventive spray guidance.",
    }

    with patch("src.graph.nodes.get_gemini_service") as mock_ggs:
        mock_ggs.return_value.generate_guidance.return_value = mock_guidance
        with open(sample_files[0], "rb") as f:
            response = client.post(
                "/analyze",
                files={"image": ("test_leaf.jpg", f, "image/jpeg")},
                data={"include_gradcam": "true", "location": "Pune"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["gradcam_base64"] is not None
    assert not data["gradcam_base64"].startswith("data:image")


def test_analyze_endpoint_invalid_file_rejects():
    """Verify POST /analyze returns 400 for non-image files before graph execution."""
    response = client.post(
        "/analyze",
        files={"image": ("invalid.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_analyze_upload_empty_file_rejected_400():
    """Verify empty 0-byte upload to /analyze returns HTTP 400 before graph execution."""
    response = client.post(
        "/analyze",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_upload_empty_file_rejected_400():
    """Verify empty 0-byte upload to /predict returns HTTP 400 before model execution."""
    response = client.post(
        "/predict",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_upload_exactly_at_limit_accepted():
    """Verify upload of exactly MAX_FILE_SIZE_BYTES (10 MB) is accepted past the size check."""
    from backend.main import MAX_FILE_SIZE_BYTES
    exact_limit_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_FILE_SIZE_BYTES - 4)
    assert len(exact_limit_bytes) == MAX_FILE_SIZE_BYTES

    response = client.post(
        "/analyze",
        files={"image": ("at_limit.jpg", exact_limit_bytes, "image/jpeg")},
    )
    # Must NOT be 413. Passes size check; fails quality check safely on synthetic noise
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error_type"] == "quality_check_failed"


def test_analyze_upload_exceeding_limit_by_one_byte_rejected_413():
    """Verify upload exceeding MAX_FILE_SIZE_BYTES by 1 byte is rejected with HTTP 413."""
    from backend.main import MAX_FILE_SIZE_BYTES
    oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_FILE_SIZE_BYTES - 3)
    assert len(oversized_bytes) == MAX_FILE_SIZE_BYTES + 1

    response = client.post(
        "/analyze",
        files={"image": ("too_large.jpg", oversized_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()


def test_oversized_upload_rejected_before_graph_execution():
    """Verify upload exceeding limit is rejected with 413 before graph workflow is invoked."""
    from unittest.mock import patch
    from backend.main import MAX_FILE_SIZE_BYTES

    oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_FILE_SIZE_BYTES - 3)
    with patch("backend.main.run_agriculture_workflow") as mock_workflow:
        response = client.post(
            "/analyze",
            files={"image": ("too_large.jpg", oversized_bytes, "image/jpeg")},
        )
        assert response.status_code == 413
        mock_workflow.assert_not_called()


def test_predict_upload_exceeding_limit_rejected_413():
    """Verify oversized upload to /predict is rejected with HTTP 413 before model inference."""
    from backend.main import MAX_FILE_SIZE_BYTES
    oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (MAX_FILE_SIZE_BYTES - 3)

    response = client.post(
        "/predict",
        files={"image": ("too_large.jpg", oversized_bytes, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()

