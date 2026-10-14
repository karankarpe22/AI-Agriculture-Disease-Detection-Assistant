"""Pydantic schemas and data contracts for the Soil Health Agent.
Strictly distinguishes field measurements, location measurements, regional estimates,
and general references without fabricating missing data.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class DataCategory(str, Enum):
    FIELD_MEASURED = "field_measured"         # Actual field/lab sample measurement
    LOCATION_MEASURED = "location_measured"   # Measured at specific coordinates (e.g., ground/sensor grid)
    REGIONAL_ESTIMATE = "regional_estimate"   # Regional, district-level, or 250m gridded modelled estimate
    GENERAL_REFERENCE = "general_reference"   # General agricultural reference benchmark
    UNAVAILABLE = "unavailable"               # Parameter not available from authorized sources


class ExtractionStatus(str, Enum):
    EXTRACTED = "extracted"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"


class SoilParameterValue(BaseModel):
    """Holds a single soil parameter with full data provenance, units, and integrity metadata."""
    parameter_name: str = Field(..., description="Canonical parameter key (e.g. ph, nitrogen, moisture)")
    display_name: str = Field(..., description="Human-readable title (e.g. Soil pH, Volumetric Moisture)")
    value: Any = Field(default=None, description="Actual numeric or descriptive value, or None if unavailable")
    unit: str | None = Field(default=None, description="Physical unit of measurement (e.g. pH, %, m3/m3, kg/ha)")
    depth: str | None = Field(default=None, description="Soil layer depth (e.g. 0-5cm, 3-9cm, root_zone)")
    source_name: str = Field(..., description="Name of the official source providing this data")
    source_url: str = Field(..., description="Authoritative URL of the data provider")
    geographic_scope: str = Field(
        default="district",
        description="Geographic scope: 'field', 'point_location', 'district', 'state', or 'national'"
    )
    observation_or_publication_date: str | None = Field(
        default=None,
        description="Observation timestamp or dataset publication period"
    )
    retrieval_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of retrieval"
    )
    data_category: DataCategory = Field(
        default=DataCategory.REGIONAL_ESTIMATE,
        description="Category: field_measured, location_measured, regional_estimate, general_reference, unavailable"
    )
    confidence_or_reliability: Literal["high", "medium", "low", "indicative", "unavailable"] = Field(
        default="medium",
        description="Reliability indicator based on source and resolution"
    )
    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.EXTRACTED,
        description="Extraction status from source"
    )
    interpretation: str | None = Field(
        default=None,
        description="Agronomic interpretation (e.g., Moderately Alkaline, Adequate Root Moisture)"
    )


class LocationDetail(BaseModel):
    """Normalized geographic location container."""
    village: str | None = Field(default=None, description="Village or local town")
    district: str = Field(..., description="District name (e.g. Pune, Nashik, Solapur)")
    state: str = Field(default="Maharashtra", description="State name")
    latitude: float | None = Field(default=None, description="Latitude coordinate")
    longitude: float | None = Field(default=None, description="Longitude coordinate")
    resolved_name: str | None = Field(default=None, description="Full geocoded location string")
    geographic_scope: str = Field(default="district", description="Effective scope: point_location, district, or state")


class DataQualityMetadata(BaseModel):
    """Metadata auditing data coverage, missing parameters, and scientific limitations."""
    overall_classification: DataCategory = Field(
        default=DataCategory.REGIONAL_ESTIMATE,
        description="Predominant category of available data"
    )
    measured_parameters_count: int = Field(default=0)
    estimated_parameters_count: int = Field(default=0)
    unavailable_parameters: list[str] = Field(
        default_factory=list,
        description="List of parameters explicitly not available in public databases"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Scientific limitations regarding field-specific variance"
    )


class SourceAttribution(BaseModel):
    """Metadata documenting each queried data source."""
    source_name: str
    organization: str
    url: str
    data_type: str
    spatial_resolution: str
    update_frequency: str
    licence: str
    retrieval_status: str


class SoilAssessment(BaseModel):
    """Interpreted assessment based on source-backed thresholds."""
    ph_status: str | None = None
    moisture_status: str | None = None
    organic_matter_status: str | None = None
    texture_type: str | None = None
    crop_suitability_notes: str | None = None


class SoilGuidance(BaseModel):
    """Bilingual agronomic guidance synthesized by Gemini and grounded in RAG evidence."""
    language: str = "english"
    summary: str = ""
    soil_condition_overview: str = ""
    crop_suitability: str = ""
    recommendations: list[str] = Field(default_factory=list)
    organic_amendments: list[str] = Field(default_factory=list)
    irrigation_advice: str | None = None
    soil_testing_advice: str = ""
    limitations_notice: str = ""


class SoilHealthRequest(BaseModel):
    """User request schema for soil health analysis."""
    location: LocationDetail | str = Field(
        ...,
        description="Location as a structured object or simple string (e.g. 'Pune' or 'Baramati, Pune')"
    )
    crop: str | None = Field(default=None, description="Optional target crop (Tomato, Potato, Bell Pepper, or general)")
    language: str = Field(default="english", description="Preferred response language: 'english' or 'marathi'")


class SoilHealthResponse(BaseModel):
    """Complete, validated soil health response contract."""
    status: Literal["success", "partial", "unavailable"] = "success"
    location: LocationDetail
    soil_parameters: dict[str, SoilParameterValue]
    soil_type: str | None = None
    data_quality: DataQualityMetadata
    sources: list[SourceAttribution]
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)
    assessment: SoilAssessment
    guidance: SoilGuidance
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
