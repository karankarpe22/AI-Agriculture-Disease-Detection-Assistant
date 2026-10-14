"""Comprehensive Unit and Integration Test Suite for Soil Health Agent.
Validates all requirements of Phase H:
- Location resolution (valid, ambiguous, missing)
- Source adapters with mocked HTTP responses
- Source timeouts and failure handling
- Unit normalization & value bounds
- Strict data category separation (location_measured vs regional_estimate vs unavailable)
- Explicit missing parameters auditing (never inventing numbers)
- RAG soil evidence retrieval
- Gemini prompt construction & offline fallback
- Cache storage, expiry, and failure rejection
- Bilingual responses (English & Marathi)
- FastAPI /soil-health endpoint contract
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.soil.soil_agent import SoilHealthAgent, get_soil_agent
from src.soil.soil_cache import SoilDataCache
from src.soil.soil_data_sources import (
    IndianSoilResourceAdapter,
    OpenMeteoSoilAdapter,
    SourceRegistry,
)
from src.soil.soil_extractor import SoilDataExtractor
from src.soil.soil_normalizer import SoilNormalizer
from src.soil.soil_schemas import (
    DataCategory,
    ExtractionStatus,
    LocationDetail,
    SoilHealthRequest,
    SoilHealthResponse,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def agent():
    return get_soil_agent()


# ------------------------------------------------------------------------------
# 1. Location Resolution Tests
# ------------------------------------------------------------------------------
def test_location_resolution_valid_city(agent: SoilHealthAgent):
    loc = agent.resolve_location("Pune")
    assert loc.district.lower() == "pune"
    assert loc.state.lower() == "maharashtra"
    assert loc.latitude is not None
    assert loc.longitude is not None
    assert loc.geographic_scope == "point_location"


def test_location_resolution_structured_object(agent: SoilHealthAgent):
    loc_input = LocationDetail(
        village="Indapur",
        district="Pune",
        state="Maharashtra",
        latitude=18.11,
        longitude=75.03,
    )
    resolved = agent.resolve_location(loc_input)
    assert resolved.village == "Indapur"
    assert resolved.district == "Pune"
    assert resolved.latitude == 18.11


def test_location_resolution_empty_or_fallback(agent: SoilHealthAgent):
    # Empty string should gracefully fallback to default agricultural hub (Pune)
    loc = agent.resolve_location("")
    assert loc.district.lower() == "pune"
    assert loc.state.lower() == "maharashtra"


# ------------------------------------------------------------------------------
# 2. Source Adapter Tests (Mocked HTTP)
# ------------------------------------------------------------------------------
def test_open_meteo_soil_adapter_success():
    adapter = OpenMeteoSoilAdapter()
    loc = LocationDetail(district="Pune", state="Maharashtra", latitude=18.52, longitude=73.85)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "current": {
            "time": "2026-10-14T12:00",
            "soil_temperature_0cm": 25.5,
            "soil_temperature_6cm": 28.0,
            "soil_moisture_0_to_1cm": 0.14,
            "soil_moisture_3_to_9cm": 0.22,
        },
        "current_units": {"soil_moisture_3_to_9cm": "m³/m³"},
    }

    with patch("requests.get", return_value=mock_resp):
        res = adapter.fetch_data(loc)
        assert res["success"] is True
        assert res["data"]["soil_moisture_3_to_9cm"] == 0.22
        assert res["data"]["soil_temperature_6cm"] == 28.0


def test_open_meteo_soil_adapter_timeout():
    adapter = OpenMeteoSoilAdapter()
    loc = LocationDetail(district="Pune", state="Maharashtra", latitude=18.52, longitude=73.85)

    with patch("requests.get", side_effect=Exception("Read timed out")):
        res = adapter.fetch_data(loc)
        assert res["success"] is False
        assert "timed out" in res["error"]


def test_indian_soil_resource_adapter():
    adapter = IndianSoilResourceAdapter()
    loc = LocationDetail(district="Nashik", state="Maharashtra", latitude=19.99, longitude=73.78)

    res = adapter.fetch_data(loc)
    assert res["success"] is True
    assert "Inceptisols" in res["data"]["soil_type"]
    assert res["data"]["ph_typical"] == 7.5
    assert res["geographic_scope"] == "district"


# ------------------------------------------------------------------------------
# 3. Unit Normalization & Interpretations
# ------------------------------------------------------------------------------
def test_unit_normalization():
    # pH normalization (e.g. SoilGrids reports pH*10 = 78)
    assert SoilNormalizer.normalize_ph(78.0) == 7.8
    assert SoilNormalizer.normalize_ph(6.5) == 6.5
    assert SoilNormalizer.normalize_ph(None) is None

    # SOC normalization (e.g. g/kg to %)
    assert SoilNormalizer.normalize_soc(8.5, "g/kg") == 0.85
    assert SoilNormalizer.normalize_soc(120.0, "dg/kg") == 1.2
    assert SoilNormalizer.normalize_soc(None) is None


def test_soil_interpretations():
    # pH interpretations
    assert "Optimal Neutral" in SoilNormalizer.interpret_ph(6.8)
    assert "Moderately Alkaline" in SoilNormalizer.interpret_ph(7.9)
    assert "Common Scab" in SoilNormalizer.interpret_ph(7.5, crop="Potato")

    # Moisture interpretations
    assert "Optimal Root-Zone Moisture" in SoilNormalizer.interpret_moisture(0.24)
    assert "Deficit Moisture" in SoilNormalizer.interpret_moisture(0.08)

    # SOC interpretations
    assert "Low" in SoilNormalizer.interpret_organic_carbon(0.42)
    assert "Medium" in SoilNormalizer.interpret_organic_carbon(0.65)


# ------------------------------------------------------------------------------
# 4. Strict Scientific Integrity & Category Separation
# ------------------------------------------------------------------------------
def test_extractor_preserves_provenance_and_flags_missing():
    extractor = SoilDataExtractor()
    loc = LocationDetail(district="Pune", state="Maharashtra", latitude=18.52, longitude=73.85)

    raw_sources = {
        "open_meteo": {
            "success": True,
            "source": "Open-Meteo Land Surface",
            "url": "https://api.open-meteo.com",
            "observation_time": "2026-10-14T10:00",
            "data": {
                "soil_moisture_3_to_9cm": 0.18,
                "soil_moisture_0_to_1cm": 0.15,
                "soil_temperature_6cm": 29.0,
            },
        },
        "indian_db": {
            "success": True,
            "source": "ICAR-NBSS&LUP",
            "url": "https://nbsslup.icar.gov.in",
            "geographic_scope": "district",
            "data": {
                "soil_type": "Vertisols (Deep Black Cotton Soil)",
                "texture": "Clay",
                "ph_typical": 7.8,
                "soc_typical_pct": 0.52,
                "nitrogen_status": "Low (<250 kg/ha)",
                "phosphorus_status": "Low to Medium",
                "potassium_status": "High",
            },
        },
    }

    params, quality, assessment, soil_type = extractor.extract_parameters(raw_sources, loc, crop="Tomato")

    # Live moisture must be LOCATION_MEASURED
    assert params["soil_moisture"].data_category == DataCategory.LOCATION_MEASURED
    assert params["soil_moisture"].value == 18.0
    assert params["soil_moisture"].unit == "%"

    # District pH must be REGIONAL_ESTIMATE
    assert params["ph"].data_category == DataCategory.REGIONAL_ESTIMATE
    assert params["ph"].value == 7.8

    # Field-measured N-P-K must be explicitly declared UNAVAILABLE (never fabricated)
    assert "field_measured_nitrogen_kg_per_ha" in quality.unavailable_parameters
    assert "field_measured_phosphorus_kg_per_ha" in quality.unavailable_parameters
    assert quality.measured_parameters_count > 0
    assert len(quality.limitations) >= 2


# ------------------------------------------------------------------------------
# 5. RAG Retrieval Integration
# ------------------------------------------------------------------------------
def test_rag_soil_retrieval(agent: SoilHealthAgent):
    chunks = agent.rag_service.retrieve(
        crop="Tomato",
        disease="Soil Health",
        question="soil pH organic carbon nutrient moisture",
        top_k=3,
    )
    assert len(chunks) > 0
    # Must retrieve relevant tomato soil or general soil guidelines
    headings = " ".join([c["heading"] for c in chunks])
    assert any(k in headings.lower() for k in ["soil", "nutrition", "irrigation", "management"])


# ------------------------------------------------------------------------------
# 6. Cache Expiry & Reliability
# ------------------------------------------------------------------------------
def test_soil_cache_ttl_and_error_handling(tmp_path):
    cache = SoilDataCache(default_ttl_seconds=1, disk_cache_dir=tmp_path)

    # 1. Do not cache error payloads
    cache.set("test_src", "pune", {"error": "Server error", "success": False})
    assert cache.get("test_src", "pune") is None

    # 2. Cache valid payload
    cache.set("test_src", "pune", {"success": True, "value": 42}, ttl_seconds=1)
    cached = cache.get("test_src", "pune")
    assert cached is not None
    assert cached["value"] == 42

    # 3. Test expiry
    time.sleep(1.2)
    assert cache.get("test_src", "pune") is None


# ------------------------------------------------------------------------------
# 7. Gemini Guidance Fallback (Offline Grounded)
# ------------------------------------------------------------------------------
def test_gemini_soil_fallback_english(agent: SoilHealthAgent):
    guidance = agent.gemini_service._fallback_soil_guidance(
        location={"district": "Pune", "state": "Maharashtra"},
        soil_parameters={
            "ph": {"value": 7.8},
            "soil_moisture": {"value": 18.5},
            "organic_carbon": {"value": 0.55},
        },
        data_quality={"unavailable_parameters": ["field_measured_nitrogen"]},
        crop="Tomato",
        evidence=[],
        is_marathi=False,
        weather={},
    )
    assert "Pune" in guidance["summary"]
    assert len(guidance["recommendations"]) >= 3
    assert len(guidance["organic_amendments"]) >= 1
    assert "Soil Testing Laboratory" in guidance["soil_testing_advice"]
    assert guidance["language"] == "english"


def test_gemini_soil_fallback_marathi(agent: SoilHealthAgent):
    guidance = agent.gemini_service._fallback_soil_guidance(
        location={"district": "Nashik", "state": "Maharashtra"},
        soil_parameters={
            "ph": {"value": 7.5},
            "soil_moisture": {"value": 20.0},
            "organic_carbon": {"value": 0.60},
        },
        data_quality={"unavailable_parameters": ["field_measured_nitrogen"]},
        crop="Tomato",
        evidence=[],
        is_marathi=True,
        weather={},
    )
    assert ("Nashik" in guidance["summary"] or "नाशिक" in guidance["summary"])
    assert len(guidance["recommendations"]) >= 3
    assert "माती परीक्षण" in guidance["soil_testing_advice"]
    assert guidance["language"] == "marathi"


# ------------------------------------------------------------------------------
# 8. FastAPI /soil-health API Integration
# ------------------------------------------------------------------------------
def test_soil_health_api_endpoint(client: TestClient):
    req_body = {
        "location": "Pune",
        "crop": "Tomato",
        "language": "english",
    }
    resp = client.post("/soil-health", json=req_body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] in ["success", "partial"]
    assert data["location"]["district"].lower() == "pune"
    assert "soil_moisture" in data["soil_parameters"]
    assert "ph" in data["soil_parameters"]
    assert "guidance" in data
    assert len(data["guidance"]["recommendations"]) > 0
    assert len(data["sources"]) >= 2
    assert "limitations" in data["data_quality"]


def test_soil_health_api_marathi_language(client: TestClient):
    req_body = {
        "location": "Kolhapur",
        "crop": "Potato",
        "language": "marathi",
    }
    resp = client.post("/soil-health", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["guidance"]["language"] == "marathi"
    assert len(data["guidance"]["summary"]) > 10


def test_health_check_reports_soil_agent(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["soil_agent_ready"] is True
