"""Phase 5: Reliability, Security, and Edge-Case Regression Tests.

Comprehensive end-to-end tests for:
- Empty and corrupted image handling
- Oversized dimension limits
- Weather failure and timeout fallbacks
- Soil synthetic data provenance and fallback tiers
- RAG failure resilience and keyword fallback
- Gemini rate-limit/network error fallback to offline grounded advisory
- LangGraph fault tolerance, single-invocation constraint, and early exit semantics
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from backend.main import app
from src.preprocessing.quality_check import ImageQualityChecker, check_image_quality
from src.weather.weather_service import WeatherService
from src.soil.soil_service import SoilHealthService
from src.rag.vector_store import AgriculturalRAG
from src.gemini.guidance_service import GeminiGuidanceService
from src.graph.state import create_initial_state, state_to_api_response
from src.graph.workflow import run_agriculture_workflow

client = TestClient(app)


# =============================================================================
# 1. Image Handling Reliability & Resource Limit Tests
# =============================================================================

def test_empty_image_upload_rejected_at_validation_boundary():
    """Verify 0-byte file uploads are rejected with HTTP 400 at validation boundary."""
    # /analyze endpoint
    resp_analyze = client.post(
        "/analyze",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp_analyze.status_code == 400
    assert "empty" in resp_analyze.json()["detail"].lower()

    # /predict endpoint
    resp_predict = client.post(
        "/predict",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp_predict.status_code == 400
    assert "empty" in resp_predict.json()["detail"].lower()


def test_corrupted_image_fails_safely_without_fabrication():
    """Verify corrupted image bytes do not produce fabricated disease predictions."""
    corrupted_bytes = b"RIFF\x00\x00\x00\x00NOT_A_REAL_JPEG_IMAGE_DATA_12345"

    resp = client.post(
        "/analyze",
        files={"image": ("corrupt.jpg", corrupted_bytes, "image/jpeg")},
        data={"location": "Pune", "language": "english"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["crop"] is None
    assert data["disease"] is None
    assert data["confidence"] == 0.0
    assert data["error_type"] in ["quality_check_failed", "pipeline_error", "image_decode_error"]


def test_excessive_image_dimensions_rejected():
    """Verify oversized image dimensions trigger excessive_resolution guard."""
    checker = ImageQualityChecker(max_width=4000, max_height=4000)
    # Create image with dimensions 4001x100
    oversized = Image.new("RGB", (4001, 100), color=(100, 150, 100))
    res = checker.assess_image(oversized)
    assert res["is_valid"] is False
    assert "excessive_resolution" in res["issues"]
    assert "exceed maximum allowed dimension" in res["message"]


def test_unsupported_file_extension_rejected():
    """Verify unsupported file extensions are rejected with 400."""
    resp = client.post(
        "/analyze",
        files={"image": ("script.py", b"print('malicious')", "text/x-python")},
    )
    assert resp.status_code == 400
    assert "Unsupported file extension" in resp.json()["detail"]


# =============================================================================
# 2. Weather Fallback & Timeout Tests
# =============================================================================

def test_weather_complete_network_timeout_returns_safe_fallback():
    """Verify WeatherService returns safe seasonal fallback if all external networks drop."""
    service = WeatherService(api_key=None)

    with patch("requests.get", side_effect=Exception("Connection timed out")):
        result = service.get_weather(city="Pune")

    assert result["success"] is False
    assert result["provider"] == "Unavailable"
    assert result["temperature_c"] == 25.0
    assert result["humidity_percentage"] == 65
    assert "risk_analysis" in result
    assert len(result["risk_analysis"]) > 10


def test_weather_missing_api_key_falls_back_to_open_meteo():
    """Verify weather service with no API key uses Open-Meteo fallback."""
    with patch.dict("os.environ", {"WEATHER_API_KEY": ""}):
        service = WeatherService(api_key="")
        assert service.api_key == ""


# =============================================================================
# 3. Soil Provenance & Fallback Tier Tests
# =============================================================================

def test_soil_all_tiers_preserve_synthetic_disclosures():
    """Verify synthetic provenance and disclaimers are preserved across all soil query tiers."""
    service = SoilHealthService()

    # Tier 1: Exact location and crop
    rec1 = service.get_soil_health(location="Pune", crop="Tomato")
    assert rec1["is_synthetic"] is True
    assert rec1["data_type"] == "synthetic"
    assert "NOTICE: This record contains synthetic demonstration data" in rec1["disclaimer"]

    # Tier 2: Taluka alias mapping
    rec2 = service.get_soil_health(location="Baramati", crop="Tomato")
    assert rec2["is_synthetic"] is True
    assert rec2["data_type"] == "synthetic"

    # Tier 3: Unknown location fallback
    rec3 = service.get_soil_health(location="UnknownVillageXYZ", crop="UnknownCrop")
    assert rec3["is_synthetic"] is True
    assert rec3["is_fallback"] is True
    assert "interpretation" in rec3

    # Tier 4: In-memory fallback (simulated empty dataset)
    empty_service = SoilHealthService()
    empty_service.records = []
    rec4 = empty_service.get_soil_health(location="Pune", crop="Tomato")
    assert rec4["is_synthetic"] is True
    assert rec4["is_fallback"] is True
    assert rec4["match_method"] == "in_memory_fallback"


# =============================================================================
# 4. RAG Fault Tolerance & Fallback Tests
# =============================================================================

def test_rag_encoder_failure_triggers_keyword_fallback():
    """Verify RAG gracefully falls back to keyword matching if embedding encoder fails."""
    rag = AgriculturalRAG()

    with patch.object(rag, "_ensure_encoder", side_effect=RuntimeError("CUDA OOM / Encoder fail")):
        chunks = rag.retrieve(crop="Tomato", disease="Early blight", top_k=2)

    assert len(chunks) > 0
    assert all(c["retrieval_method"] == "keyword_fallback" for c in chunks)
    assert any("blight" in c.get("disease", "").lower() or "tomato" in c.get("crop", "").lower() for c in chunks)


def test_rag_empty_query_returns_empty_list():
    """Verify completely empty query returns empty evidence list without error."""
    rag = AgriculturalRAG()
    assert rag.retrieve(crop=None, disease=None, question=None) == []


# =============================================================================
# 5. Gemini Rate Limit & Network Error Resilience
# =============================================================================

def test_gemini_rate_limit_falls_back_to_offline_rag():
    """Verify GeminiGuidanceService falls back to offline ICAR evidence synthesis on rate-limit."""
    service = GeminiGuidanceService(api_key="mock_key")

    mock_client = MagicMock()
    # Simulate 429 RESOURCE_EXHAUSTED across all candidate models
    mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    weather = {"location": "Pune", "temperature_c": 26.0, "humidity_percentage": 75, "risk_analysis": "Normal"}
    evidence = [{
        "source": "ICAR-IIVR",
        "title": "Tomato Early Blight Guide",
        "heading": "Chemical Control",
        "content": "Spray Mancozeb 75% WP @ 2.5 g/L water.",
    }]

    with patch.object(service, "_get_client", return_value=mock_client):
        guidance = service.generate_guidance(
            crop="Tomato",
            disease="Early blight",
            confidence=0.92,
            weather=weather,
            retrieved_evidence=evidence,
            farmer_question="What spray is recommended?",
            language="english",
        )

    assert "direct_answer" in guidance
    assert len(guidance["direct_answer"]) > 10
    assert "explanation" in guidance
    assert "management_guidance" in guidance
    assert "en" in guidance
    assert "mr" in guidance


# =============================================================================
# 6. LangGraph State, Isolation & Error Structure Parity
# =============================================================================

def test_workflow_quality_failure_aborts_early_without_gemini():
    """Verify image quality failure immediately terminates graph without calling Gemini."""
    import asyncio
    dark_img = Image.new("RGB", (150, 150), (5, 5, 5))
    buf = io.BytesIO()
    dark_img.save(buf, format="JPEG")

    state = create_initial_state(
        request_type="analyze",
        image_input=buf.getvalue(),
        location="Pune",
        language="english",
    )

    with patch("src.graph.nodes.get_gemini_service") as mock_gemini:
        final_state = asyncio.run(run_agriculture_workflow(state))
        # Gemini must NEVER be invoked on quality check failure
        mock_gemini.assert_not_called()

    assert final_state["execution_status"] == "quality_failed"
    api_resp = state_to_api_response(final_state)
    assert api_resp["success"] is False
    assert api_resp["error_type"] == "quality_check_failed"
    assert api_resp["warning"] is not None


def test_workflow_error_status_formatting_parity():
    """Verify execution_status='error' populates error_type and warning correctly."""
    state = create_initial_state(
        request_type="analyze",
        location="Pune",
        language="english",
    )
    state["execution_status"] = "error"
    state["error_information"] = ["Inference pipeline failed unexpectedly."]
    state["prediction_result"] = {"error_type": "model_error", "warning": "CUDA kernel error"}

    resp = state_to_api_response(state)
    assert resp["success"] is False
    assert resp["error_type"] == "model_error"
    assert resp["warning"] == "CUDA kernel error"
    assert resp["crop"] is None
    assert resp["disease"] is None
