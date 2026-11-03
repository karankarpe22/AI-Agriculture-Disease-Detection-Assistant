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
    assert result["match_method"] in ["regional_baseline", "regional_benchmark", "fallback"]
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


# ==============================================================================
# Comprehensive Deterministic Matching & Diagnostic Tests (12 Scenarios)
# ==============================================================================


def test_matching_exact_location_crop():
    """Scenario 1: Exact normalized location + exact crop match."""
    service = get_soil_service()
    result = service.get_soil_health(location="Pune", crop="Tomato")
    assert result["record_id"] == "SYN-SOIL-MH-PUN-001"
    assert result["match_method"] == "exact_location_crop"
    assert result["is_fallback"] is False
    assert result["location"] == "Pune"
    assert result["crop_type"] == "Tomato"
    assert result["is_synthetic"] is True


def test_matching_exact_location_general():
    """Scenario 2: Exact normalized location + General crop fallback."""
    service = get_soil_service()
    # Dapoli in Ratnagiri only has General in the synthetic dataset
    result = service.get_soil_health(location="Dapoli", crop="Tomato")
    assert result["record_id"] == "SYN-SOIL-MH-RAT-003"
    assert result["match_method"] == "exact_location_general"
    assert result["is_fallback"] is False
    assert result["location"] == "Dapoli"
    assert result["crop_type"] == "General"


def test_matching_district_crop():
    """Scenario 3: District-level exact crop match (sub-basin alias / taluka)."""
    service = get_soil_service()
    # Pimpri is an alias to Pune district; Pune has Potato record SYN-SOIL-MH-PUN-002
    result = service.get_soil_health(location="Pimpri", crop="Potato")
    assert result["district"] == "Pune"
    assert result["crop_type"] == "Potato"
    assert result["match_method"] == "exact_district_crop"
    assert result["is_fallback"] is True
    assert result["record_id"] == "SYN-SOIL-MH-PUN-002"


def test_matching_district_general():
    """Scenario 4: District-level General crop fallback."""
    service = get_soil_service()
    # Pimpri is in Pune; Pune district does not have Sugarcane, falls back to Pune district General record
    result = service.get_soil_health(location="Pimpri", crop="Sugarcane")
    assert result["district"] == "Pune"
    assert result["crop_type"] == "General"
    assert result["match_method"] == "exact_district_general"
    assert result["is_fallback"] is True
    assert result["record_id"] == "SYN-SOIL-MH-AKU-002"


def test_matching_unknown_location_baseline():
    """Scenario 5: Unknown location regional baseline fallback."""
    service = get_soil_service()
    result = service.get_soil_health(location="HimalayanValleyXYZ", crop="Tomato")
    assert result["record_id"] == "SYN-SOIL-MH-DEFAULT-001"
    assert result["match_method"] == "regional_baseline"
    assert result["is_fallback"] is True
    assert result["is_synthetic"] is True


def test_matching_missing_location():
    """Scenario 6: Missing location (None or empty)."""
    service = get_soil_service()
    result_none = service.get_soil_health(location=None, crop="Tomato")
    assert result_none["record_id"] == "SYN-SOIL-MH-DEFAULT-001"
    assert result_none["match_method"] == "regional_baseline"
    assert result_none["is_fallback"] is True

    result_empty = service.get_soil_health(location="", crop="Tomato")
    assert result_empty["record_id"] == "SYN-SOIL-MH-DEFAULT-001"
    assert result_empty["match_method"] == "regional_baseline"
    assert result_empty["is_fallback"] is True


def test_matching_missing_crop():
    """Scenario 7: Missing crop (None or empty)."""
    service = get_soil_service()
    # Case A: Location has a General record (e.g. Akurdi) -> exact_location_general (is_fallback=False)
    result_akurdi = service.get_soil_health(location="Akurdi", crop=None)
    assert result_akurdi["record_id"] == "SYN-SOIL-MH-AKU-002"
    assert result_akurdi["match_method"] == "exact_location_general"
    assert result_akurdi["is_fallback"] is False

    # Case B: Location has specific crops but no General (e.g. Pune) -> falls back to district General (is_fallback=True)
    result_pune = service.get_soil_health(location="Pune", crop=None)
    assert result_pune["district"] == "Pune"
    assert result_pune["crop_type"] == "General"
    assert result_pune["match_method"] == "exact_district_general"
    assert result_pune["is_fallback"] is True

    # Empty string crop
    result_empty = service.get_soil_health(location="Akurdi", crop="")
    assert result_empty["record_id"] == "SYN-SOIL-MH-AKU-002"
    assert result_empty["match_method"] == "exact_location_general"
    assert result_empty["is_fallback"] is False


def test_matching_crop_normalization():
    """Scenario 8: Crop name normalization, trimming, and case-insensitivity."""
    service = get_soil_service()
    # Case & whitespace
    r1 = service.get_soil_health(location="  puNE  ", crop="  TOMATO  ")
    assert r1["record_id"] == "SYN-SOIL-MH-PUN-001"
    assert r1["match_method"] == "exact_location_crop"

    # Crop synonym: 'capsicum' -> 'bell pepper'
    r2 = service.get_soil_health(location="Nashik", crop="capsicum")
    assert r2["record_id"] == "SYN-SOIL-MH-NAS-002"
    assert r2["crop_type"] == "Bell pepper"
    assert r2["match_method"] == "exact_location_crop"

    # Crop synonym: 'pepper, bell' -> 'bell pepper'
    r3 = service.get_soil_health(location="Nashik", crop="pepper, bell")
    assert r3["record_id"] == "SYN-SOIL-MH-NAS-002"
    assert r3["match_method"] == "exact_location_crop"


def test_matching_no_accidental_substring_crop():
    """Scenario 9: Prevent accidental substring matching between unrelated crops."""
    service = get_soil_service()
    # 'pot' must not accidentally match 'potato'
    result_pot = service.get_soil_health(location="Pune", crop="pot")
    assert result_pot["match_method"] != "exact_location_crop"
    assert result_pot["crop_type"] == "General"

    # 'to' must not accidentally match 'tomato'
    result_to = service.get_soil_health(location="Pune", crop="to")
    assert result_to["match_method"] != "exact_location_crop"
    assert result_to["crop_type"] == "General"


def test_matching_deterministic_tie_breaking():
    """Scenario 10: Deterministic tie-breaking across multiple calls."""
    service = get_soil_service()
    results = [service.get_soil_health(location="Pune", crop="Tomato") for _ in range(5)]
    ids = [r["record_id"] for r in results]
    assert all(rid == "SYN-SOIL-MH-PUN-001" for rid in ids)

    # District query with multiple possible matches resolves deterministically
    district_results = [service.get_soil_health(location="pune", crop="Potato") for _ in range(5)]
    district_ids = [r["record_id"] for r in district_results]
    assert all(rid == district_ids[0] for rid in district_ids)


def test_matching_diagnostic_metadata_fields():
    """Scenario 11: Verification of transparent diagnostic metadata fields."""
    service = get_soil_service()
    result = service.get_soil_health(location="Kolhapur", crop="Sugarcane")
    assert "match_method" in result
    assert "is_fallback" in result
    assert "queried_location" in result
    assert "queried_crop" in result
    assert result["queried_location"] == "Kolhapur"
    assert result["queried_crop"] == "Sugarcane"
    assert result["match_method"] in [
        "exact_location_crop",
        "exact_location_general",
        "exact_district_crop",
        "exact_district_general",
        "regional_baseline",
        "in_memory_fallback",
    ]
    assert isinstance(result["is_fallback"], bool)


def test_matching_api_compatibility():
    """Scenario 12: Compatibility with existing API responses and synthetic data fields."""
    service = get_soil_service()
    result = service.get_soil_health(location="Pune", crop="Tomato")

    # Required top-level fields
    expected_fields = [
        "record_id",
        "location",
        "district",
        "agro_climatic_zone",
        "crop_type",
        "soil_type",
        "soil_ph",
        "soil_moisture_percent",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "organic_carbon_percent",
        "electrical_conductivity_ds_m",
        "drainage_class",
        "data_type",
        "dataset_version",
        "demonstration_purpose",
        "disclaimer",
        "units",
        "is_synthetic",
        "is_fallback",
        "match_method",
        "interpretation",
    ]
    for field in expected_fields:
        assert field in result, f"Missing required field: {field}"

    # Synthetic disclosures
    assert result["data_type"] == "synthetic"
    assert result["dataset_version"] == "1.0.0-synthetic-demo"
    assert result["is_synthetic"] is True
    assert "synthetic" in result["disclaimer"].lower()

    # Interpretation schema
    interp = result["interpretation"]
    interp_fields = [
        "ph_status",
        "ph_status_mr",
        "moisture_status",
        "moisture_status_mr",
        "nitrogen_rating",
        "nitrogen_rating_mr",
        "phosphorus_rating",
        "phosphorus_rating_mr",
        "potassium_rating",
        "potassium_rating_mr",
        "organic_carbon_rating",
        "disease_soil_interaction",
        "disease_soil_interaction_mr",
    ]
    for ifield in interp_fields:
        assert ifield in interp, f"Missing interpretation field: {ifield}"
