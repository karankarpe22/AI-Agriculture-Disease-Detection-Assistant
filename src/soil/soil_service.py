"""Soil Health Agent module for synthetic agro-climatic soil profiling.

Provides structured, simulated soil health parameters (pH, moisture, N-P-K, organic carbon)
grounded in regional baseline profiles for academic demonstration. All records are
explicitly labeled as synthetic demonstration data.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("src.soil.soil_service")

DATASET_VERSION = "1.0.0-synthetic-demo"
DEFAULT_DATASET_PATH = Path("data/soil/synthetic_soil_records.json")

# Regional sub-basin and taluka alias mapping to parent districts
ALIAS_MAP: dict[str, str] = {
    "akurdi": "pune",
    "pimpri": "pune",
    "chinchwad": "pune",
    "baramati": "pune",
    "hadapsar": "pune",
    "niphad": "nashik",
    "malegaon": "nashik",
    "dindori": "nashik",
    "pandharpur": "solapur",
    "barshi": "solapur",
    "katol": "nagpur",
    "saoner": "nagpur",
    "karveer": "kolhapur",
    "gadhinglaj": "kolhapur",
}

# Canonical crop normalization to eliminate accidental substring collisions
CROP_SYNONYMS: dict[str, str] = {
    "tomato": "tomato",
    "potato": "potato",
    "bell pepper": "bell pepper",
    "pepper, bell": "bell pepper",
    "pepper": "bell pepper",
    "capsicum": "bell pepper",
    "general": "general",
    "general / solanaceae": "general",
    "all": "general",
    "onion": "onion",
    "cotton": "cotton",
    "soybean": "soybean",
    "soya": "soybean",
    "rice": "rice",
    "paddy": "rice",
    "sugarcane": "sugarcane",
}


def _normalize_text(val: str | None) -> str:
    """Normalize text by stripping whitespace and converting to lowercase."""
    if not val:
        return ""
    return re.sub(r"\s+", " ", str(val).strip().lower())


def _normalize_crop(crop: str | None) -> str:
    """Normalize crop name to canonical form to prevent accidental substring matching."""
    cleaned = _normalize_text(crop)
    if not cleaned:
        return ""
    return CROP_SYNONYMS.get(cleaned, cleaned)


def _crops_match(user_crop: str, record_crop: str) -> bool:
    """Check if normalized crops match exactly, avoiding accidental substring collisions."""
    u = _normalize_crop(user_crop)
    r = _normalize_crop(record_crop)
    if not u or not r:
        return False
    return u == r


UNITS = {
    "soil_ph": "pH scale (0-14)",
    "soil_moisture_percent": "percentage (%)",
    "nitrogen_kg_ha": "kilograms per hectare (kg/ha)",
    "phosphorus_kg_ha": "kilograms per hectare (kg/ha)",
    "potassium_kg_ha": "kilograms per hectare (kg/ha)",
    "organic_carbon_percent": "percentage (%)",
    "electrical_conductivity_ds_m": "deciSiemens per metre (dS/m)",
}

SYNTHETIC_DISCLAIMER = (
    "NOTICE: This record contains synthetic demonstration data generated for academic illustration. "
    "It must not be used as an actual laboratory soil test report or for physical fertilizer prescription."
)


class SoilHealthRecord(BaseModel):
    """Validated schema for synthetic soil health records."""

    record_id: str
    location: str
    district: str
    agro_climatic_zone: str
    crop_type: str = "General"
    soil_type: str
    soil_ph: float = Field(..., ge=3.5, le=10.5)
    soil_moisture_percent: float = Field(..., ge=0.0, le=100.0)
    nitrogen_kg_ha: float = Field(..., ge=0.0, le=1000.0)
    phosphorus_kg_ha: float = Field(..., ge=0.0, le=500.0)
    potassium_kg_ha: float = Field(..., ge=0.0, le=1500.0)
    organic_carbon_percent: float = Field(..., ge=0.0, le=5.0)
    electrical_conductivity_ds_m: float = Field(default=0.4, ge=0.0, le=10.0)
    drainage_class: str = "Moderately well-drained"
    data_type: str = "synthetic"
    dataset_version: str = DATASET_VERSION
    demonstration_purpose: str = "Academic demonstration only"
    disclaimer: str = SYNTHETIC_DISCLAIMER
    units: dict[str, str] = Field(default_factory=lambda: dict(UNITS))
    is_synthetic: bool = True
    is_fallback: bool = False
    match_method: str = "exact"

    @field_validator("data_type")
    @classmethod
    def ensure_synthetic_data_type(cls, v: str) -> str:
        if v.lower() != "synthetic":
            raise ValueError("All records in this module must be explicitly tagged as 'synthetic'.")
        return "synthetic"


class SoilHealthService:
    """Service retrieving, validating, and interpreting synthetic soil health records."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        self.dataset_path = Path(dataset_path or DEFAULT_DATASET_PATH)
        self.records: list[dict[str, Any]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load and validate synthetic soil records from JSON dataset."""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_records = data.get("records", [])
                    validated = []
                    for r in raw_records:
                        rec = SoilHealthRecord(**r)
                        validated.append(rec.model_dump())
                    self.records = validated
                    logger.info(f"Loaded {len(self.records)} synthetic soil records from {self.dataset_path}")
                    return
            except Exception as e:
                logger.warning(f"Failed to load synthetic soil records from {self.dataset_path}: {e}")

        # In-memory default benchmark fallback if file cannot be read
        self.records = [self._create_default_benchmark().model_dump()]

    def _create_default_benchmark(self) -> SoilHealthRecord:
        """Create standard regional Deccan Vertisol benchmark record."""
        return SoilHealthRecord(
            record_id="SYN-SOIL-MH-DEFAULT-001",
            location="Maharashtra Regional Baseline",
            district="General Maharashtra Agricultural Region",
            agro_climatic_zone="Deccan Plateau Agro-Climatic Baseline",
            crop_type="General",
            soil_type="Standard Deccan Vertisol (Medium Black Agricultural Soil)",
            soil_ph=7.0,
            soil_moisture_percent=25.0,
            nitrogen_kg_ha=180.0,
            phosphorus_kg_ha=22.0,
            potassium_kg_ha=300.0,
            organic_carbon_percent=0.55,
            electrical_conductivity_ds_m=0.40,
            drainage_class="Moderately well-drained",
            is_fallback=True,
            match_method="in_memory_fallback",
        )

    def get_soil_health(
        self,
        location: str | None = None,
        crop: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve suitable synthetic soil record matching location and crop, with fallback.
        
        Deterministic Matching Priority:
        1. Exact normalized location + exact crop (match_method: 'exact_location_crop', is_fallback: False)
        2. Exact normalized location + General crop (match_method: 'exact_location_general', is_fallback: False)
        3. Exact normalized district + exact crop (match_method: 'exact_district_crop', is_fallback: True)
        4. Exact normalized district + General crop (match_method: 'exact_district_general', is_fallback: True)
        5. Statewide synthetic baseline (match_method: 'regional_baseline', is_fallback: True)
        6. In-memory fallback if dataset cannot be loaded (match_method: 'in_memory_fallback', is_fallback: True)
        """
        loc_clean = _normalize_text(location)
        crop_clean = _normalize_crop(crop)
        alias_district = ALIAS_MAP.get(loc_clean, "")

        district_targets: set[str] = set()
        if loc_clean:
            district_targets.add(loc_clean)
        if alias_district:
            district_targets.add(alias_district)

        matched_record: dict[str, Any] | None = None
        match_method: str = "regional_baseline"
        is_fallback: bool = True

        # Check if records are loaded
        if not self.records:
            benchmark = self._create_default_benchmark().model_dump()
            benchmark["match_method"] = "in_memory_fallback"
            benchmark["is_fallback"] = True
            benchmark["queried_location"] = location or ""
            benchmark["queried_crop"] = crop or ""
            benchmark["interpretation"] = self._interpret_soil_parameters(benchmark)
            return benchmark

        # 1. Exact normalized location + exact crop
        if loc_clean and crop_clean and crop_clean != "general":
            cands = [
                r for r in self.records
                if _normalize_text(r.get("location")) == loc_clean and _crops_match(crop_clean, r.get("crop_type"))
            ]
            if cands:
                cands.sort(key=lambda x: str(x.get("record_id", "")))
                matched_record = dict(cands[0])
                match_method = "exact_location_crop"
                is_fallback = False

        # 2. Exact normalized location + General crop
        if not matched_record and loc_clean:
            cands = [
                r for r in self.records
                if _normalize_text(r.get("location")) == loc_clean and _normalize_crop(r.get("crop_type")) == "general"
            ]
            if cands:
                cands.sort(key=lambda x: str(x.get("record_id", "")))
                matched_record = dict(cands[0])
                match_method = "exact_location_general"
                is_fallback = False

        # 3. Exact normalized district + exact crop
        if not matched_record and district_targets and crop_clean and crop_clean != "general":
            cands = [
                r for r in self.records
                if _normalize_text(r.get("district")) in district_targets and _crops_match(crop_clean, r.get("crop_type"))
            ]
            if cands:
                # Deterministic sorting: prefer records whose location matches target, then by record_id
                cands.sort(key=lambda x: (
                    0 if _normalize_text(x.get("location")) in district_targets else 1,
                    str(x.get("record_id", ""))
                ))
                matched_record = dict(cands[0])
                match_method = "exact_district_crop"
                is_fallback = True

        # 4. Exact normalized district + General crop
        if not matched_record and district_targets:
            cands = [
                r for r in self.records
                if _normalize_text(r.get("district")) in district_targets and _normalize_crop(r.get("crop_type")) == "general"
            ]
            if cands:
                cands.sort(key=lambda x: (
                    0 if _normalize_text(x.get("location")) in district_targets else 1,
                    str(x.get("record_id", ""))
                ))
                matched_record = dict(cands[0])
                match_method = "exact_district_general"
                is_fallback = True

        # 5. Statewide synthetic baseline
        if not matched_record:
            baseline_cands = [
                r for r in self.records
                if r.get("record_id") == "SYN-SOIL-MH-DEFAULT-001" or "default" in str(r.get("record_id", "")).lower()
            ]
            if baseline_cands:
                matched_record = dict(baseline_cands[0])
                match_method = "regional_baseline"
                is_fallback = True
            else:
                # 6. In-memory fallback if dataset is empty or baseline record missing
                matched_record = self._create_default_benchmark().model_dump()
                match_method = "in_memory_fallback"
                is_fallback = True

        # Attach transparent diagnostic metadata
        matched_record["match_method"] = match_method
        matched_record["is_fallback"] = is_fallback
        matched_record["queried_location"] = location or ""
        matched_record["queried_crop"] = crop or ""

        # Generate agronomic interpretation
        interpretation = self._interpret_soil_parameters(matched_record)
        matched_record["interpretation"] = interpretation

        return matched_record

    def _interpret_soil_parameters(self, record: dict[str, Any]) -> dict[str, Any]:
        """Generate agronomic interpretation for synthetic soil parameters."""
        ph = record.get("soil_ph", 7.0)
        moisture = record.get("soil_moisture_percent", 25.0)
        n = record.get("nitrogen_kg_ha", 180.0)
        p = record.get("phosphorus_kg_ha", 22.0)
        k = record.get("potassium_kg_ha", 300.0)
        oc = record.get("organic_carbon_percent", 0.55)

        # pH Status
        if ph < 6.0:
            ph_status = "Moderately Acidic"
            ph_status_mr = "मध्यम आम्लधर्मी"
        elif 6.0 <= ph <= 7.5:
            ph_status = "Optimal Neutral"
            ph_status_mr = "उत्कृष्ट उदासीन (योग्य)"
        elif 7.5 < ph <= 8.2:
            ph_status = "Slightly Alkaline / Calcareous"
            ph_status_mr = "किंचित अल्कधर्मी (चुनखडीयुक्त)"
        else:
            ph_status = "Strongly Alkaline"
            ph_status_mr = "तीव्र अल्कधर्मी"

        # Moisture Status
        if moisture < 20.0:
            moisture_status = "Low (Moisture Stress Potential)"
            moisture_status_mr = "कमी ओलावा (पाण्याचा ताण)"
        elif 20.0 <= moisture <= 30.0:
            moisture_status = "Optimal Field Capacity"
            moisture_status_mr = "योग्य वाफसा स्थिती"
        else:
            moisture_status = "High / Saturated (Aeration Risk)"
            moisture_status_mr = "अति ओलावा (हवा खेळती राहण्यात अडचण)"

        # NPK Rating
        n_rating = "Low" if n < 200 else ("Medium" if n <= 350 else "High")
        p_rating = "Low" if p < 20 else ("Medium" if p <= 40 else "High")
        k_rating = "Low" if k < 150 else ("Medium" if k <= 280 else "High")

        n_rating_mr = "कमी" if n_rating == "Low" else ("मध्यम" if n_rating == "Medium" else "मुबलक")
        p_rating_mr = "कमी" if p_rating == "Low" else ("मध्यम" if p_rating == "Medium" else "मुबलक")
        k_rating_mr = "कमी" if k_rating == "Low" else ("मध्यम" if k_rating == "Medium" else "मुबलक")

        # Tripartite Disease-Soil Interaction Note
        notes = []
        if moisture > 28.0:
            notes.append(
                "Elevated soil moisture combined with humid microclimates creates saturated root zones "
                "conducive to fungal zoospore motility and soil-borne damping-off / late blight progression."
            )
        elif moisture < 18.0:
            notes.append(
                "Sub-optimal soil moisture induces root drought stress, reducing calcium translocation "
                "and making plant tissue more vulnerable to stress-related necrosis and opportunistic mites."
            )

        if n > 220.0:
            notes.append(
                "High available nitrogen promotes rapid, succulent foliar vegetative growth with thinner "
                "epidermal cell walls that are more readily penetrated by fungal haustoria."
            )
        elif k >= 280.0:
            notes.append(
                "Adequate potassium levels support cell wall lignification, enhancing foliar resilience "
                "against pathogen penetration."
            )

        interaction_summary = " ".join(notes) or (
            "Soil parameters are within regional equilibrium with moderate disease vulnerability."
        )

        # Marathi summary
        notes_mr = []
        if moisture > 28.0:
            notes_mr.append(
                "मातीतील जास्त ओलावा आणि हवेतील आर्द्रता यामुळे मुळांभोवती पाणी साचून बुरशीजन्य रोग व करपा "
                "पसरण्यासाठी पोषक वातावरण तयार होते."
            )
        elif moisture < 18.0:
            notes_mr.append(
                "मातीत कमी ओलावा असल्याने पिकावर पाण्याचा ताण येतो आणि झाडांची नैसर्गिक रोगप्रतिकारक शक्ती घटते."
            )
        if k >= 280.0:
            notes_mr.append("जमिनीतील पोटॅशचे चांगले प्रमाण पानांच्या पेशींना बळकटी देऊन रोगाशी लढण्यास मदत करते.")

        interaction_summary_mr = " ".join(notes_mr) or "जमिनीतील गुणधर्म संतुलित असून पीक पोषणासाठी अनुकूल आहेत."

        return {
            "ph_status": ph_status,
            "ph_status_mr": ph_status_mr,
            "moisture_status": moisture_status,
            "moisture_status_mr": moisture_status_mr,
            "nitrogen_rating": n_rating,
            "nitrogen_rating_mr": n_rating_mr,
            "phosphorus_rating": p_rating,
            "phosphorus_rating_mr": p_rating_mr,
            "potassium_rating": k_rating,
            "potassium_rating_mr": k_rating_mr,
            "organic_carbon_rating": "Medium" if 0.50 <= oc <= 0.75 else ("High" if oc > 0.75 else "Low"),
            "soil_type_mr": self._translate_soil_type(record.get("soil_type", "")),
            "disease_soil_interaction": interaction_summary,
            "disease_soil_interaction_mr": interaction_summary_mr,
        }

    def _translate_soil_type(self, soil_type: str) -> str:
        """Translate soil type description to Marathi."""
        st = soil_type.lower()
        if "black" in st or "vertisol" in st:
            return "मध्यम काळी कसदार जमीन (काळी माती / मध्यम भारी)"
        if "sandy" in st:
            return "वाळूमिश्रित पोयटा जमीन (हलकी ते मध्यम)"
        if "lateritic" in st or "red" in st:
            return "तांबडी जांभी जमीन (उत्कृष्ट निचरा)"
        return "काळी कसदार शेतीची जमीन"


# Default singleton
_default_soil_service: SoilHealthService | None = None


def get_soil_service() -> SoilHealthService:
    """Singleton getter for SoilHealthService."""
    global _default_soil_service
    if _default_soil_service is None:
        _default_soil_service = SoilHealthService()
    return _default_soil_service
