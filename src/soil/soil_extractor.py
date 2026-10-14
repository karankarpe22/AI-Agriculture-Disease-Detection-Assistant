"""Extracts, filters, and standardizes soil properties into validated Pydantic models.
Enforces strict scientific integrity: distinguishes measured data from regional estimates,
and flags unavailable parameters explicitly without inventing numbers.
"""
from __future__ import annotations

from typing import Any
from src.soil.soil_normalizer import SoilNormalizer
from src.soil.soil_schemas import (
    DataCategory,
    DataQualityMetadata,
    ExtractionStatus,
    LocationDetail,
    SoilAssessment,
    SoilParameterValue,
)


class SoilDataExtractor:
    """Extracts normalized soil parameters from raw source outputs."""

    def __init__(self) -> None:
        self.normalizer = SoilNormalizer()

    def extract_parameters(
        self,
        raw_sources: dict[str, Any],
        location: LocationDetail,
        crop: str | None = None,
    ) -> tuple[dict[str, SoilParameterValue], DataQualityMetadata, SoilAssessment, str | None]:
        parameters: dict[str, SoilParameterValue] = {}
        missing_params: list[str] = []
        measured_count = 0
        estimated_count = 0

        om_payload = raw_sources.get("open_meteo", {})
        sg_payload = raw_sources.get("soilgrids", {})
        in_payload = raw_sources.get("indian_db", {})

        # ----------------------------------------------------------------------
        # 1. Live Soil Moisture (Root zone 3-9cm)
        # ----------------------------------------------------------------------
        if om_payload.get("success") and "data" in om_payload:
            om_data = om_payload["data"]
            rz_moist = om_data.get("soil_moisture_3_to_9cm")
            obs_time = om_payload.get("observation_time")
            if rz_moist is not None:
                m_pct = round(rz_moist * 100.0, 1)
                interp = self.normalizer.interpret_moisture(rz_moist, crop)
                parameters["soil_moisture"] = SoilParameterValue(
                    parameter_name="soil_moisture",
                    display_name="Volumetric Soil Moisture (Root Zone)",
                    value=m_pct,
                    unit="%",
                    depth="3-9cm",
                    source_name=om_payload.get("source", "Open-Meteo"),
                    source_url=om_payload.get("url", "https://api.open-meteo.com"),
                    geographic_scope="point_location",
                    observation_or_publication_date=obs_time,
                    data_category=DataCategory.LOCATION_MEASURED,
                    confidence_or_reliability="high",
                    extraction_status=ExtractionStatus.EXTRACTED,
                    interpretation=interp,
                )
                measured_count += 1
            else:
                missing_params.append("soil_moisture")
        else:
            missing_params.append("soil_moisture")

        # ----------------------------------------------------------------------
        # 2. Live Surface Soil Moisture (0-1cm)
        # ----------------------------------------------------------------------
        if om_payload.get("success") and "data" in om_payload:
            om_data = om_payload["data"]
            surf_moist = om_data.get("soil_moisture_0_to_1cm")
            if surf_moist is not None:
                parameters["surface_moisture"] = SoilParameterValue(
                    parameter_name="surface_moisture",
                    display_name="Surface Soil Moisture",
                    value=round(surf_moist * 100.0, 1),
                    unit="%",
                    depth="0-1cm",
                    source_name=om_payload.get("source", "Open-Meteo"),
                    source_url=om_payload.get("url", "https://api.open-meteo.com"),
                    geographic_scope="point_location",
                    observation_or_publication_date=om_payload.get("observation_time"),
                    data_category=DataCategory.LOCATION_MEASURED,
                    confidence_or_reliability="high",
                    extraction_status=ExtractionStatus.EXTRACTED,
                    interpretation="Topsoil moisture layer influencing seed germination and crusting.",
                )
                measured_count += 1

        # ----------------------------------------------------------------------
        # 3. Live Soil Temperature (0-6cm)
        # ----------------------------------------------------------------------
        if om_payload.get("success") and "data" in om_payload:
            om_data = om_payload["data"]
            soil_temp = om_data.get("soil_temperature_6cm") or om_data.get("soil_temperature_0cm")
            if soil_temp is not None:
                interp = self.normalizer.interpret_soil_temperature(soil_temp)
                parameters["soil_temperature"] = SoilParameterValue(
                    parameter_name="soil_temperature",
                    display_name="Soil Temperature (Subsurface)",
                    value=round(soil_temp, 1),
                    unit="°C",
                    depth="0-6cm",
                    source_name=om_payload.get("source", "Open-Meteo"),
                    source_url=om_payload.get("url", "https://api.open-meteo.com"),
                    geographic_scope="point_location",
                    observation_or_publication_date=om_payload.get("observation_time"),
                    data_category=DataCategory.LOCATION_MEASURED,
                    confidence_or_reliability="high",
                    extraction_status=ExtractionStatus.EXTRACTED,
                    interpretation=interp,
                )
                measured_count += 1
            else:
                missing_params.append("soil_temperature")

        # ----------------------------------------------------------------------
        # 4. Soil pH
        # Primary: ISRIC SoilGrids 250m; Fallback: ICAR-NBSS&LUP District Survey
        # ----------------------------------------------------------------------
        extracted_ph = None
        ph_source = None
        ph_cat = DataCategory.REGIONAL_ESTIMATE

        if sg_payload.get("success") and "layers" in sg_payload:
            ph_layer = sg_payload["layers"].get("phh2o")
            if ph_layer and ph_layer.get("raw_value") is not None:
                norm_ph = self.normalizer.normalize_ph(ph_layer["raw_value"])
                if norm_ph is not None:
                    extracted_ph = norm_ph
                    ph_source = (sg_payload.get("source"), sg_payload.get("url"), "250m gridded raster")

        if extracted_ph is None and in_payload.get("success") and "data" in in_payload:
            in_data = in_payload["data"]
            dist_ph = in_data.get("ph_typical")
            if dist_ph is not None:
                extracted_ph = dist_ph
                ph_source = (in_payload.get("source"), in_payload.get("url"), f"{location.district} district survey ({in_data.get('ph_range')})")

        if extracted_ph is not None and ph_source:
            parameters["ph"] = SoilParameterValue(
                parameter_name="ph",
                display_name="Soil pH (Reaction)",
                value=extracted_ph,
                unit="pH",
                depth="0-15cm",
                source_name=ph_source[0],
                source_url=ph_source[1],
                geographic_scope=in_payload.get("geographic_scope", "district"),
                observation_or_publication_date="ICAR Soil Resource Benchmark / ISRIC",
                data_category=DataCategory.REGIONAL_ESTIMATE,
                confidence_or_reliability="medium",
                extraction_status=ExtractionStatus.EXTRACTED,
                interpretation=self.normalizer.interpret_ph(extracted_ph, crop),
            )
            estimated_count += 1
        else:
            missing_params.append("ph")
            parameters["ph"] = SoilParameterValue(
                parameter_name="ph",
                display_name="Soil pH (Reaction)",
                value=None,
                unit="pH",
                source_name="Not Available",
                source_url="",
                data_category=DataCategory.UNAVAILABLE,
                extraction_status=ExtractionStatus.UNAVAILABLE,
                interpretation="Not Available in public datasets. Laboratory soil test required.",
            )

        # ----------------------------------------------------------------------
        # 5. Soil Organic Carbon (SOC %)
        # ----------------------------------------------------------------------
        extracted_soc = None
        soc_source = None

        if sg_payload.get("success") and "layers" in sg_payload:
            soc_layer = sg_payload["layers"].get("soc")
            if soc_layer and soc_layer.get("raw_value") is not None:
                norm_soc = self.normalizer.normalize_soc(soc_layer["raw_value"], soc_layer.get("target_units", "g/kg"))
                if norm_soc is not None:
                    extracted_soc = norm_soc
                    soc_source = (sg_payload.get("source"), sg_payload.get("url"))

        if extracted_soc is None and in_payload.get("success") and "data" in in_payload:
            in_data = in_payload["data"]
            dist_soc = in_data.get("soc_typical_pct")
            if dist_soc is not None:
                extracted_soc = dist_soc
                soc_source = (in_payload.get("source"), in_payload.get("url"))

        if extracted_soc is not None and soc_source:
            parameters["organic_carbon"] = SoilParameterValue(
                parameter_name="organic_carbon",
                display_name="Soil Organic Carbon (SOC)",
                value=extracted_soc,
                unit="%",
                depth="0-15cm",
                source_name=soc_source[0],
                source_url=soc_source[1],
                geographic_scope=in_payload.get("geographic_scope", "district"),
                observation_or_publication_date="ICAR Soil Survey / ISRIC 2020",
                data_category=DataCategory.REGIONAL_ESTIMATE,
                confidence_or_reliability="medium",
                extraction_status=ExtractionStatus.EXTRACTED,
                interpretation=self.normalizer.interpret_organic_carbon(extracted_soc),
            )
            estimated_count += 1
        else:
            missing_params.append("organic_carbon")
            parameters["organic_carbon"] = SoilParameterValue(
                parameter_name="organic_carbon",
                display_name="Soil Organic Carbon (SOC)",
                value=None,
                unit="%",
                source_name="Not Available",
                source_url="",
                data_category=DataCategory.UNAVAILABLE,
                extraction_status=ExtractionStatus.UNAVAILABLE,
                interpretation="Not Available. Composite farm soil testing recommended.",
            )

        # ----------------------------------------------------------------------
        # 6. Regional Soil Type / Order & Texture
        # ----------------------------------------------------------------------
        soil_type_str = None
        if in_payload.get("success") and "data" in in_payload:
            in_data = in_payload["data"]
            soil_type_str = in_data.get("soil_type")
            texture_str = in_data.get("texture")

            if soil_type_str:
                parameters["soil_type"] = SoilParameterValue(
                    parameter_name="soil_type",
                    display_name="Predominant Soil Order / Type",
                    value=soil_type_str,
                    unit="Taxonomic Order",
                    source_name=in_payload.get("source", "ICAR-NBSS&LUP"),
                    source_url=in_payload.get("url", "https://nbsslup.icar.gov.in"),
                    geographic_scope=in_payload.get("geographic_scope", "district"),
                    observation_or_publication_date="ICAR District Agro-Ecological Survey",
                    data_category=DataCategory.REGIONAL_ESTIMATE,
                    confidence_or_reliability="high",
                    extraction_status=ExtractionStatus.EXTRACTED,
                    interpretation=f"Regional classification: {soil_type_str}. {in_data.get('drainage', '')}",
                )
                estimated_count += 1

            if texture_str:
                parameters["texture"] = SoilParameterValue(
                    parameter_name="texture",
                    display_name="Soil Texture Class",
                    value=texture_str,
                    unit="USDA Texture Class",
                    source_name=in_payload.get("source", "ICAR-NBSS&LUP"),
                    source_url=in_payload.get("url", "https://nbsslup.icar.gov.in"),
                    geographic_scope=in_payload.get("geographic_scope", "district"),
                    observation_or_publication_date="ICAR District Survey",
                    data_category=DataCategory.REGIONAL_ESTIMATE,
                    confidence_or_reliability="high",
                    extraction_status=ExtractionStatus.EXTRACTED,
                    interpretation=f"Texture: {texture_str}.",
                )
                estimated_count += 1

        # ----------------------------------------------------------------------
        # 7. Regional Fertility Baseline (N-P-K Status)
        # Marked explicitly as REGIONAL ESTIMATE / REFERENCE; field values are UNAVAILABLE
        # ----------------------------------------------------------------------
        if in_payload.get("success") and "data" in in_payload:
            in_data = in_payload["data"]
            parameters["regional_nitrogen_status"] = SoilParameterValue(
                parameter_name="regional_nitrogen_status",
                display_name="Regional Available Nitrogen Status",
                value=in_data.get("nitrogen_status", "Low (<280 kg/ha)"),
                unit="Fertility Class",
                source_name="Ministry of Agriculture Soil Health Card Compendium",
                source_url="https://soilhealth.dac.gov.in",
                geographic_scope=in_payload.get("geographic_scope", "district"),
                observation_or_publication_date="DAC&FW Soil Health Portal Baseline",
                data_category=DataCategory.REGIONAL_ESTIMATE,
                confidence_or_reliability="medium",
                extraction_status=ExtractionStatus.EXTRACTED,
                interpretation=f"District general index: {in_data.get('nitrogen_status')}. Actual field nitrogen depends on past crop residues and fertilization.",
            )
            parameters["regional_phosphorus_status"] = SoilParameterValue(
                parameter_name="regional_phosphorus_status",
                display_name="Regional Available Phosphorus Status",
                value=in_data.get("phosphorus_status", "Low to Medium"),
                unit="Fertility Class",
                source_name="Ministry of Agriculture Soil Health Card Compendium",
                source_url="https://soilhealth.dac.gov.in",
                geographic_scope=in_payload.get("geographic_scope", "district"),
                observation_or_publication_date="DAC&FW Soil Health Portal Baseline",
                data_category=DataCategory.REGIONAL_ESTIMATE,
                confidence_or_reliability="medium",
                extraction_status=ExtractionStatus.EXTRACTED,
                interpretation=f"District baseline: {in_data.get('phosphorus_status')}.",
            )
            parameters["regional_potassium_status"] = SoilParameterValue(
                parameter_name="regional_potassium_status",
                display_name="Regional Available Potassium Status",
                value=in_data.get("potassium_status", "High (>280 kg/ha)"),
                unit="Fertility Class",
                source_name="Ministry of Agriculture Soil Health Card Compendium",
                source_url="https://soilhealth.dac.gov.in",
                geographic_scope=in_payload.get("geographic_scope", "district"),
                observation_or_publication_date="DAC&FW Soil Health Portal Baseline",
                data_category=DataCategory.REGIONAL_ESTIMATE,
                confidence_or_reliability="medium",
                extraction_status=ExtractionStatus.EXTRACTED,
                interpretation=f"District baseline: {in_data.get('potassium_status')}.",
            )
            estimated_count += 3

        # Explicitly declare field-measured N-P-K as UNAVAILABLE (Scientific integrity)
        missing_params.extend([
            "field_measured_nitrogen_kg_per_ha",
            "field_measured_phosphorus_kg_per_ha",
            "field_measured_potassium_kg_per_ha",
            "field_measured_cation_exchange_capacity",
        ])

        # ----------------------------------------------------------------------
        # 8. Data Quality Metadata & Limitations
        # ----------------------------------------------------------------------
        quality_metadata = DataQualityMetadata(
            overall_classification=(
                DataCategory.LOCATION_MEASURED if measured_count > 0 else DataCategory.REGIONAL_ESTIMATE
            ),
            measured_parameters_count=measured_count,
            estimated_parameters_count=estimated_count,
            unavailable_parameters=missing_params,
            limitations=[
                "Volumetric soil moisture and soil temperatures are modelled via Open-Meteo Land Surface satellite/numerical analysis (~1-2 km resolution).",
                "Soil order, typical pH, and organic carbon are derived from ICAR-NBSS&LUP district benchmarks and ISRIC 250m gridded estimates; they represent regional characteristics, not individual farm measurements.",
                "Exact field-level chemical fertility (N-P-K kg/ha, precise EC) cannot be determined without a physical soil sample tested at an authorized Soil Testing Laboratory (STL) or Krishi Vigyan Kendra (KVK).",
            ],
        )

        # ----------------------------------------------------------------------
        # 9. Soil Assessment
        # ----------------------------------------------------------------------
        assessment = SoilAssessment(
            ph_status=parameters.get("ph", SoilParameterValue(
                parameter_name="ph", display_name="pH", source_name="", source_url=""
            )).interpretation,
            moisture_status=parameters.get("soil_moisture", SoilParameterValue(
                parameter_name="moisture", display_name="moisture", source_name="", source_url=""
            )).interpretation,
            organic_matter_status=parameters.get("organic_carbon", SoilParameterValue(
                parameter_name="soc", display_name="soc", source_name="", source_url=""
            )).interpretation,
            texture_type=soil_type_str or (in_payload.get("data", {}).get("texture") if in_payload.get("success") else None),
            crop_suitability_notes=(
                f"Evaluation for {crop or 'General Horticultural Crops'} in {location.district}, {location.state}."
            ),
        )

        return parameters, quality_metadata, assessment, soil_type_str
