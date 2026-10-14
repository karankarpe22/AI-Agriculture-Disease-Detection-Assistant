"""Central Soil Health Agent Orchestrator.
Coordinates:
1. Location resolution and geographic scope determination.
2. Verified multi-source soil data acquisition (Open-Meteo Land Surface, ISRIC SoilGrids, ICAR-NBSS&LUP).
3. Data extraction, unit normalization, and strict data category auditing.
4. RAG knowledge retrieval from curated ICAR/TNAU agricultural documents.
5. Evidence-grounded synthesis via Google Gemini with multi-model fallback.
6. Caching, defensive timeouts, and scientific integrity (no invented values).
"""
from __future__ import annotations

import logging
from typing import Any
import requests

from src.gemini.guidance_service import get_gemini_service
from src.rag.vector_store import get_rag_service
from src.soil.soil_cache import get_soil_cache
from src.soil.soil_data_sources import SourceRegistry
from src.soil.soil_extractor import SoilDataExtractor
from src.soil.soil_schemas import (
    DataCategory,
    DataQualityMetadata,
    ExtractionStatus,
    LocationDetail,
    SoilAssessment,
    SoilGuidance,
    SoilHealthRequest,
    SoilHealthResponse,
    SoilParameterValue,
)
from src.weather.weather_service import get_weather_service

logger = logging.getLogger("soil.agent")


class SoilHealthAgent:
    """Orchestrator for location-driven, evidence-grounded soil health intelligence."""

    def __init__(self) -> None:
        self.cache = get_soil_cache()
        self.source_registry = SourceRegistry()
        self.extractor = SoilDataExtractor()
        self.rag_service = get_rag_service()
        self.gemini_service = get_gemini_service()
        self.weather_service = get_weather_service()

    def resolve_location(self, location_input: LocationDetail | str | dict[str, Any]) -> LocationDetail:
        """Resolve arbitrary user location string or model into normalized LocationDetail with coordinates."""
        village = None
        district = "Pune"
        state = "Maharashtra"
        lat = None
        lon = None
        resolved_name = None

        if isinstance(location_input, LocationDetail):
            return location_input
        elif isinstance(location_input, dict):
            village = location_input.get("village")
            district = location_input.get("district") or location_input.get("city") or "Pune"
            state = location_input.get("state") or "Maharashtra"
            lat = location_input.get("latitude")
            lon = location_input.get("longitude")
        elif isinstance(location_input, str):
            clean_str = location_input.strip()
            if not clean_str:
                district = "Pune"
            elif "," in clean_str:
                parts = [p.strip() for p in clean_str.split(",") if p.strip()]
                if len(parts) >= 3:
                    village = parts[0]
                    district = parts[1]
                    state = parts[2]
                elif len(parts) == 2:
                    district = parts[0]
                    state = parts[1]
                else:
                    district = parts[0]
            else:
                district = clean_str

        # If coordinates missing, geocode via Open-Meteo keyless geocoding
        if lat is None or lon is None:
            query = f"{village}, {district}, {state}" if village else f"{district}, {state}"
            try:
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
                resp = requests.get(geo_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        lat = results[0].get("latitude")
                        lon = results[0].get("longitude")
                        resolved_name = f"{results[0].get('name')}, {results[0].get('admin1', state)}, {results[0].get('country', 'India')}"
            except Exception as e:
                logger.warning(f"Geocoding note for '{query}': {e}")

        # Fallback coordinates for common Maharashtra agricultural districts
        if lat is None or lon is None:
            coord_map = {
                "pune": (18.52, 73.85),
                "nashik": (19.99, 73.78),
                "solapur": (17.65, 75.90),
                "kolhapur": (16.70, 74.24),
                "nagpur": (21.14, 79.08),
                "ahmednagar": (19.09, 74.74),
                "satara": (17.68, 73.99),
                "aurangabad": (19.87, 75.34),
                "chhatrapati sambhajinagar": (19.87, 75.34),
                "amravati": (20.93, 77.75),
                "sangli": (16.85, 74.58),
            }
            mapped = coord_map.get(district.lower().strip())
            if mapped:
                lat, lon = mapped
            else:
                # Default to Pune district coordinates
                lat, lon = (18.52, 73.85)

        return LocationDetail(
            village=village,
            district=district.capitalize(),
            state=state.capitalize(),
            latitude=lat,
            longitude=lon,
            resolved_name=resolved_name or f"{district.capitalize()}, {state.capitalize()}",
            geographic_scope="point_location" if (lat and lon) else "district",
        )

    def analyze_soil_health(self, request: SoilHealthRequest) -> SoilHealthResponse:
        """Execute end-to-end soil health assessment and agronomic guidance generation."""
        # 1. Resolve Location
        location = self.resolve_location(request.location)
        loc_cache_key = f"{location.district}_{location.state}_{location.latitude}_{location.longitude}"

        # 2. Check Cache
        cached_data = self.cache.get("soil_agent", loc_cache_key, params_key=request.crop or "general")
        if cached_data:
            logger.info(f"Serving cached soil health assessment for {loc_cache_key}")
            return SoilHealthResponse(**cached_data)

        # 3. Query Verified Soil Sources
        query_res = self.source_registry.query_all_sources(location)
        raw_sources = query_res["raw_results"]
        attributions = query_res["attributions"]

        # 4. Extract and Audit Parameters
        soil_parameters, quality_meta, assessment, soil_type = self.extractor.extract_parameters(
            raw_sources=raw_sources,
            location=location,
            crop=request.crop,
        )

        # 5. Retrieve RAG Evidence from Curated ICAR/TNAU Documents
        crop_name = request.crop or "General"
        rag_query = f"soil pH organic carbon nutrient moisture management {crop_name}"
        retrieved_chunks = self.rag_service.retrieve(
            crop=request.crop if request.crop and request.crop.lower() in ["tomato", "potato", "pepper", "bell pepper"] else None,
            disease="Soil Health",
            question=rag_query,
            top_k=3,
        )

        # 6. Retrieve Live Weather Microclimate Context
        weather_context: dict[str, Any] = {}
        try:
            weather_context = self.weather_service.get_weather(
                city=location.district,
                latitude=location.latitude,
                longitude=location.longitude,
            )
        except Exception as e:
            logger.warning(f"Weather context note: {e}")

        # 7. Generate Grounded Gemini Guidance
        soil_params_dict = {k: v.model_dump() for k, v in soil_parameters.items()}
        guidance_dict = self.gemini_service.generate_soil_guidance(
            location=location.model_dump(),
            soil_parameters=soil_params_dict,
            data_quality=quality_meta.model_dump(),
            crop=request.crop,
            retrieved_evidence=retrieved_chunks,
            language=request.language,
            weather_context=weather_context,
        )

        guidance = SoilGuidance(
            language=request.language,
            summary=guidance_dict.get("summary", ""),
            soil_condition_overview=guidance_dict.get("soil_condition_overview", ""),
            crop_suitability=guidance_dict.get("crop_suitability", ""),
            recommendations=guidance_dict.get("recommendations", []),
            organic_amendments=guidance_dict.get("organic_amendments", []),
            irrigation_advice=guidance_dict.get("irrigation_advice"),
            soil_testing_advice=guidance_dict.get("soil_testing_advice", ""),
            limitations_notice=guidance_dict.get("limitations_notice", ""),
        )

        # 8. Determine Overall Status
        status = "success"
        if quality_meta.measured_parameters_count == 0:
            status = "partial"

        response = SoilHealthResponse(
            status=status,
            location=location,
            soil_parameters=soil_parameters,
            soil_type=soil_type,
            data_quality=quality_meta,
            sources=attributions,
            retrieved_evidence=retrieved_chunks,
            assessment=assessment,
            guidance=guidance,
        )

        # 9. Cache Validated Result (TTL: 1 hour for live soil data)
        self.cache.set(
            source_name="soil_agent",
            location_key=loc_cache_key,
            data=response.model_dump(),
            params_key=request.crop or "general",
            ttl_seconds=3600,
        )

        return response


_default_agent: SoilHealthAgent | None = None


def get_soil_agent() -> SoilHealthAgent:
    """Singleton getter for SoilHealthAgent."""
    global _default_agent
    if _default_agent is None:
        _default_agent = SoilHealthAgent()
    return _default_agent
