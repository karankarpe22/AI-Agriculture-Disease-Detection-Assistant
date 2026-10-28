"""Unit tests for Soil Health Agent and synthetic soil dataset."""
from __future__ import annotations

import json
from pathlib import Path
import pytest
from src.soil.soil_service import SoilHealthService, SoilHealthRecord, get_soil_service
from src.gemini.guidance_service import get_gemini_service


def test_synthetic_soil_dataset_exists():
    """Verify synthetic soil dataset files exist on disk with valid formatting."""
    json_path = Path("data/soil/synthetic_soil_records.json")
    csv_path = Path("data/soil/synthetic_soil_records.csv")
    assert json_path.exists(), "synthetic_soil_records.json must exist"
    assert csv_path.exists(), "synthetic_soil_records.csv must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "records" in data
    assert len(data["records"]) >= 10
    assert data["data_type"] == "synthetic"
    assert "units" in data


def test_soil_record_schema_validation():
    """Verify all records conform strictly to SoilHealthRecord schema."""
    service = get_soil_service()
    assert len(service.records) >= 10

    for rec_dict in service.records:
        rec = SoilHealthRecord(**rec_dict)
        assert rec.is_synthetic is True
        assert rec.data_type == "synthetic"
        assert 3.5 <= rec.soil_ph <= 10.5
        assert 0.0 <= rec.soil_moisture_percent <= 100.0
        assert rec.nitrogen_kg_ha >= 0.0
        assert rec.phosphorus_kg_ha >= 0.0
        assert rec.potassium_kg_ha >= 0.0
        assert rec.organic_carbon_percent >= 0.0
        assert "pH scale" in rec.units.get("soil_ph", "")
        assert "kg/ha" in rec.units.get("nitrogen_kg_ha", "")


def test_soil_matching_exact():
    """Test exact location and crop matching."""
    service = get_soil_service()
    result = service.get_soil_health(location="Pune", crop="Tomato")
    assert result["location"] == "Pune"
    assert result["crop_type"] == "Tomato"
    assert result["match_method"] == "exact_location_crop"
    assert result["is_synthetic"] is True
    assert result["is_fallback"] is False
    assert "interpretation" in result
    assert "Optimal Neutral" in result["interpretation"]["ph_status"]


def test_soil_matching_sub_basin_alias():
    """Test regional sub-basin matching (Akurdi in Pune district)."""
    service = get_soil_service()
    result = service.get_soil_health(location="Akurdi", crop="Potato")
    assert result["location"] in ["Akurdi", "Pune"]
    assert result["crop_type"] == "Potato"
    assert result["is_synthetic"] is True
    assert "moisture_status" in result["interpretation"]


def test_soil_matching_fallback_unknown_location():
    """Test graceful fallback for unknown location without inventing arbitrary values."""
    service = get_soil_service()
    result = service.get_soil_health(location="UnknownHimalayanVillage", crop="Tomato")
    assert result["is_fallback"] is True
    assert result["match_method"] in ["regional_benchmark", "fallback"]
    assert result["is_synthetic"] is True
    assert "SYN-SOIL-MH-DEFAULT" in result["record_id"]
    assert "interpretation" in result


def test_soil_matching_missing_location():
    """Test behavior when location is None or empty."""
    service = get_soil_service()
    result = service.get_soil_health(location=None, crop=None)
    assert result["is_fallback"] is True
    assert result["is_synthetic"] is True
    assert result["soil_ph"] > 0
    assert "interpretation" in result


def test_gemini_tripartite_integration_with_soil():
    """Test that Gemini guidance service accepts soil context and generates integrated advisory."""
    gemini = get_gemini_service()
    soil_service = get_soil_service()
    soil_data = soil_service.get_soil_health(location="Pune", crop="Potato")

    weather_mock = {
        "location": "Pune",
        "temperature_c": 22.0,
        "humidity_percentage": 82,
        "condition": "Humid / Overcast",
        "risk_analysis": "High humidity favors late blight spore germination.",
    }
    evidence_mock = [
        {
            "source": "ICAR Potato Advisory",
            "title": "Late Blight Management",
            "heading": "Chemical Control",
            "content": "Apply Mancozeb 75% WP @ 2.5 g/L preventively. Rogue infected lower leaves.",
        }
    ]

    guidance = gemini.generate_guidance(
        crop="Potato",
        disease="Late blight",
        confidence=0.94,
        weather=weather_mock,
        retrieved_evidence=evidence_mock,
        farmer_question="What care should be taken to prevent this disease?",
        language="english",
        is_low_confidence=False,
        soil=soil_data,
    )

    assert "direct_answer" in guidance
    assert "soil_interpretation" in guidance
    assert "en" in guidance
    assert "mr" in guidance
    assert "soil_interpretation" in guidance["en"]
    assert "soil_interpretation" in guidance["mr"]
    assert "Synthetic Demonstration Data" in guidance["soil_interpretation"] or "synthetic" in guidance["soil_interpretation"].lower()
    # Check that direct_answer synthesizes crop, weather, and soil
    assert "Mancozeb" in guidance["direct_answer"] or "earthing-up" in guidance["direct_answer"] or "moisture" in guidance["direct_answer"].lower()
