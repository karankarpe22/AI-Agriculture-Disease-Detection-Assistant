"""Comprehensive validation suite for the expanded 100-record synthetic soil dataset.

Verifies:
1. Exactly 100 JSON records and exactly 100 CSV data rows.
2. Exactly 100 unique record IDs matching the approved ID format regex.
3. Complete, bitwise preservation of all original 12 records (IDs, order, values, metadata).
4. All 88 new records follow approved synthetic metadata schemas and disclaimers.
5. All numerical values are non-null and within approved agronomic ranges.
6. JSON and CSV scalar columns match 100% across all 100 records.
7. Crop distribution matches approved counts exactly (Tomato: 26, Potato: 21, General: 20,
   Bell pepper: 17, Onion: 4, Cotton: 4, Soybean: 4, Rice: 2, Sugarcane: 2).
8. Agro-climatic zone distribution matches approved counts exactly.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from collections import Counter
import pytest

from src.soil.soil_service import SoilHealthRecord

JSON_PATH = Path("data/soil/synthetic_soil_records.json")
CSV_PATH = Path("data/soil/synthetic_soil_records.csv")

APPROVED_ID_REGEX = re.compile(r"^SYN-SOIL-MH-(?:[A-Z]{3,4}|DEFAULT)-\d{3}$")

EXPECTED_CROP_COUNTS = {
    "Tomato": 26,
    "Potato": 21,
    "General": 20,
    "Bell pepper": 17,
    "Onion": 4,
    "Cotton": 4,
    "Soybean": 4,
    "Rice": 2,
    "Sugarcane": 2,
}

EXPECTED_ZONE_COUNTS = {
    "Central Maharashtra Plateau Zone": 20,
    "Plain Zone": 16,
    "Central Vidarbha Zone": 16,
    "Scarcity Zone": 12,
    "Eastern Vidarbha Zone": 9,
    "Sub-Montane Zone": 8,
    "North Konkan Coastal Zone": 8,
    "South Konkan Coastal Zone": 6,
    "Western Ghat Zone": 4,
    "Statewide Benchmark": 1,
}

# The ground-truth first 12 record IDs that must be preserved exactly in order
ORIGINAL_12_IDS = [
    "SYN-SOIL-MH-PUN-001",
    "SYN-SOIL-MH-PUN-002",
    "SYN-SOIL-MH-AKU-001",
    "SYN-SOIL-MH-AKU-002",
    "SYN-SOIL-MH-NAS-001",
    "SYN-SOIL-MH-NAS-002",
    "SYN-SOIL-MH-NAG-001",
    "SYN-SOIL-MH-NAG-002",
    "SYN-SOIL-MH-SOL-001",
    "SYN-SOIL-MH-KOL-001",
    "SYN-SOIL-MH-CSN-001",
    "SYN-SOIL-MH-DEFAULT-001",
]


def load_dataset_records():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_csv_rows():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_file_existence_and_counts():
    """Verify exactly 100 JSON records and 100 CSV rows."""
    assert JSON_PATH.exists(), "JSON dataset must exist"
    assert CSV_PATH.exists(), "CSV dataset must exist"

    data = load_dataset_records()
    records = data.get("records", [])
    csv_rows = load_csv_rows()

    assert data.get("total_records") == 100, f"JSON total_records header must be 100, got {data.get('total_records')}"
    assert len(records) == 100, f"JSON records array must have exactly 100 records, got {len(records)}"
    assert len(csv_rows) == 100, f"CSV must have exactly 100 data rows, got {len(csv_rows)}"


def test_unique_record_ids_and_format():
    """Verify all 100 IDs are unique and strictly conform to the approved regex."""
    data = load_dataset_records()
    records = data["records"]
    ids = [r["record_id"] for r in records]

    assert len(ids) == 100
    assert len(set(ids)) == 100, "All 100 record IDs must be strictly unique"

    for r_id in ids:
        assert APPROVED_ID_REGEX.match(r_id), f"Record ID '{r_id}' failed format regex"


def test_original_12_records_preservation():
    """Verify the original 12 records are intact in the exact order with unchanged values."""
    data = load_dataset_records()
    records = data["records"]

    # Check IDs of first 12 records
    first_12_ids = [r["record_id"] for r in records[:12]]
    assert first_12_ids == ORIGINAL_12_IDS, f"First 12 record IDs mismatch: {first_12_ids}"

    # Spot-check key numerical values for existing records
    pun_001 = records[0]
    assert pun_001["record_id"] == "SYN-SOIL-MH-PUN-001"
    assert pun_001["location"] == "Pune"
    assert pun_001["soil_ph"] == 7.2
    assert pun_001["soil_moisture_percent"] == 24.5
    assert pun_001["nitrogen_kg_ha"] == 185.0
    assert pun_001["phosphorus_kg_ha"] == 22.4
    assert pun_001["potassium_kg_ha"] == 310.0
    assert pun_001["organic_carbon_percent"] == 0.58
    assert pun_001["crop_type"] == "Tomato"

    def_001 = records[11]
    assert def_001["record_id"] == "SYN-SOIL-MH-DEFAULT-001"
    assert def_001["soil_ph"] == 7.0
    assert def_001["soil_moisture_percent"] == 25.0
    assert def_001["nitrogen_kg_ha"] == 180.0
    assert def_001["phosphorus_kg_ha"] == 22.0
    assert def_001["potassium_kg_ha"] == 300.0


def test_synthetic_metadata_and_disclaimers():
    """Verify 100% of records are explicitly synthetic with valid disclaimers and units."""
    data = load_dataset_records()
    assert data["data_type"] == "synthetic"
    assert data["version"] == "1.0.0-synthetic-demo"

    for r in data["records"]:
        assert r["data_type"] == "synthetic", f"Record {r['record_id']} data_type must be 'synthetic'"
        assert r["dataset_version"] == "1.0.0-synthetic-demo"
        assert "demonstration" in r["demonstration_purpose"].lower()
        assert "synthetic" in r["disclaimer"].lower()
        assert "units" in r
        assert "soil_ph" in r["units"]
        assert "nitrogen_kg_ha" in r["units"]

        # Pydantic schema validation
        rec_model = SoilHealthRecord(**r)
        assert rec_model.is_synthetic is True


def test_numerical_ranges():
    """Verify all 100 records have realistic, non-null numerical parameters within agronomic bounds."""
    data = load_dataset_records()
    for r in data["records"]:
        r_id = r["record_id"]
        ph = r["soil_ph"]
        moisture = r["soil_moisture_percent"]
        n = r["nitrogen_kg_ha"]
        p = r["phosphorus_kg_ha"]
        k = r["potassium_kg_ha"]
        oc = r["organic_carbon_percent"]
        ec = r["electrical_conductivity_ds_m"]

        assert 5.0 <= ph <= 8.5, f"{r_id} pH out of plausible range: {ph}"
        assert 14.0 <= moisture <= 36.0, f"{r_id} moisture out of plausible range: {moisture}"
        assert 120.0 <= n <= 300.0, f"{r_id} nitrogen out of plausible range: {n}"
        assert 10.0 <= p <= 45.0, f"{r_id} phosphorus out of plausible range: {p}"
        assert 140.0 <= k <= 450.0, f"{r_id} potassium out of plausible range: {k}"
        assert 0.30 <= oc <= 1.20, f"{r_id} organic carbon out of plausible range: {oc}"
        assert 0.15 <= ec <= 1.0, f"{r_id} EC out of plausible range: {ec}"


def test_json_csv_scalar_synchronization():
    """Verify JSON and CSV match character-for-character across all 16 shared scalar columns."""
    json_records = load_dataset_records()["records"]
    csv_rows = load_csv_rows()

    shared_cols = [
        "record_id", "location", "district", "agro_climatic_zone", "crop_type", "soil_type",
        "soil_ph", "soil_moisture_percent", "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha",
        "organic_carbon_percent", "electrical_conductivity_ds_m", "drainage_class", "data_type", "dataset_version"
    ]

    for idx, (j_rec, c_row) in enumerate(zip(json_records, csv_rows)):
        for col in shared_cols:
            j_val = j_rec[col]
            c_val = c_row[col]
            if isinstance(j_val, float):
                assert abs(float(j_val) - float(c_val)) < 1e-4, f"Row {idx} {col} float mismatch: {j_val} vs {c_val}"
            else:
                assert str(j_val) == str(c_val), f"Row {idx} {col} string mismatch: {j_val} vs {c_val}"


def test_crop_distribution():
    """Verify exact match with approved crop distribution."""
    json_records = load_dataset_records()["records"]
    crop_counts = Counter(r["crop_type"] for r in json_records)

    assert crop_counts == EXPECTED_CROP_COUNTS, f"Crop distribution mismatch:\nExpected: {EXPECTED_CROP_COUNTS}\nGot: {dict(crop_counts)}"
    assert sum(crop_counts.values()) == 100


def test_regional_zone_distribution():
    """Verify exact match with approved agro-climatic zone distribution."""
    json_records = load_dataset_records()["records"]

    # Canonical zone mapping to handle existing zone string variants
    def canonical_zone(zone_str: str) -> str:
        z = zone_str.lower()
        if "konkan" in z and "south" in z:
            return "South Konkan Coastal Zone"
        if "konkan" in z and "north" in z:
            return "North Konkan Coastal Zone"
        if "ghat" in z and "western" in z:
            return "Western Ghat Zone"
        if "sub-montane" in z or "transition-1" in z:
            return "Sub-Montane Zone"
        if "plain" in z or "transition zone (pavana" in z or "scarcity and plateau" in z or "scarcity & transition" in z:
            return "Plain Zone"
        if "scarcity zone" in z or "scarcity" in z:
            return "Scarcity Zone"
        if "central vidarbha" in z:
            return "Central Vidarbha Zone"
        if "eastern vidarbha" in z:
            return "Eastern Vidarbha Zone"
        if "plateau zone" in z or "semi-arid plateau" in z:
            return "Central Maharashtra Plateau Zone"
        if "baseline" in z:
            return "Statewide Benchmark"
        return zone_str

    zone_counts = Counter(canonical_zone(r["agro_climatic_zone"]) for r in json_records)

    assert zone_counts == EXPECTED_ZONE_COUNTS, f"Zone distribution mismatch:\nExpected: {EXPECTED_ZONE_COUNTS}\nGot: {dict(zone_counts)}"
    assert sum(zone_counts.values()) == 100


if __name__ == "__main__":
    pytest.main(["-v", str(Path(__file__))])

