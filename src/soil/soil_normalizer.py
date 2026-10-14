"""Soil parameter normalizer and agronomic threshold interpreter.
Standardizes units, parameter keys, and ranges across heterogeneous data sources.
Applies source-backed agronomic thresholds (ICAR, TNAU, FAO) without inventing values.
"""
from __future__ import annotations

from typing import Any
from src.soil.soil_schemas import DataCategory, SoilParameterValue


class SoilNormalizer:
    """Provides unit conversions and scientific interpretations for soil properties."""

    @staticmethod
    def normalize_ph(val: float | None, raw_unit: str = "-") -> float | None:
        """ISRIC SoilGrids reports pH*10; standard scale is 0 to 14."""
        if val is None:
            return None
        # SoilGrids d_factor of 10
        if val > 14.0 and val <= 140.0:
            return round(val / 10.0, 2)
        if 0.0 <= val <= 14.0:
            return round(float(val), 2)
        return None

    @staticmethod
    def normalize_soc(val: float | None, unit: str = "g/kg") -> float | None:
        """Normalize soil organic carbon to percentage (%)."""
        if val is None:
            return None
        # If in dg/kg (decigram/kg) or g/kg: 10 g/kg = 1.0%
        if "g/kg" in unit or "dg/kg" in unit:
            # 1 g/kg = 0.1%
            if val > 100:  # e.g. dg/kg
                return round(val / 100.0, 2)
            return round(val / 10.0, 2)
        if 0.0 <= val <= 20.0:
            return round(float(val), 2)
        return None

    @staticmethod
    def interpret_ph(ph: float | None, crop: str | None = None) -> str:
        """Provide scientific interpretation of soil pH considering crop tolerance."""
        if ph is None:
            return "Not Available"

        crop_lower = (crop or "").lower()
        if ph < 5.5:
            base = "Strongly Acidic (Al/Fe toxicity risk; liming recommended)"
        elif 5.5 <= ph < 6.2:
            base = "Moderately Acidic"
        elif 6.2 <= ph <= 7.5:
            base = "Optimal Neutral (Favorable for nutrient bioavailability)"
        elif 7.5 < ph <= 8.4:
            base = "Moderately Alkaline (Typical of Deccan Vertisols; monitor micronutrient uptake)"
        else:
            base = "Strongly Alkaline / Calcareous (High P fixation risk; gypsum/organic buffering needed)"

        # Crop-specific adaptations
        if "potato" in crop_lower:
            if ph < 5.2:
                return f"{base} — Low for potato; lime moderately to reach pH 5.5."
            elif 5.2 <= ph <= 6.4:
                return f"{base} — Ideal acidic range for potato (minimizes Common Scab incidence)."
            elif ph > 6.8:
                return f"{base} — Alkaline for potato; elevated risk of Common Scab (Streptomyces scabies)."
        elif "tomato" in crop_lower:
            if 6.0 <= ph <= 7.0:
                return f"{base} — Ideal pH range for tomato fruit set and balanced calcium uptake."
            elif ph > 7.8:
                return f"{base} — High pH for tomato; supplement chelated iron/zinc and ensure steady calcium."
        elif "pepper" in crop_lower or "capsicum" in crop_lower:
            if 6.0 <= ph <= 6.8:
                return f"{base} — Ideal range for bell pepper root development and blossom health."

        return base

    @staticmethod
    def interpret_moisture(vol_moisture: float | None, crop: str | None = None) -> str:
        """Interpret volumetric soil moisture (m3/m3 or %) in root feeding zone (0-9cm)."""
        if vol_moisture is None:
            return "Not Available"

        # Normalize to m3/m3 if expressed as percentage
        m = vol_moisture / 100.0 if vol_moisture > 1.0 else vol_moisture
        pct = m * 100.0

        crop_lower = (crop or "").lower()
        if m < 0.10:
            status = f"Deficit Moisture ({pct:.1f}% vol) — Crop drought stress; irrigation required."
        elif 0.10 <= m < 0.18:
            status = f"Moderate Moisture ({pct:.1f}% vol) — Adequate for maintenance; monitor topsoil."
        elif 0.18 <= m <= 0.35:
            status = f"Optimal Root-Zone Moisture ({pct:.1f}% vol) — Ideal for vegetative growth and nutrient uptake."
        else:
            status = f"High Moisture / Saturated ({pct:.1f}% vol) — Elevated risk of poor soil aeration and root rot."

        if "pepper" in crop_lower and m > 0.32:
            return f"{status} Note: Bell pepper roots are highly sensitive to waterlogging."
        if "tomato" in crop_lower and m < 0.12:
            return f"{status} Warning: Irregular moisture triggers Blossom End Rot (calcium deficiency)."

        return status

    @staticmethod
    def interpret_organic_carbon(soc_pct: float | None) -> str:
        """Interpret Soil Organic Carbon (SOC) status based on ICAR Soil Health Card limits."""
        if soc_pct is None:
            return "Not Available"
        if soc_pct < 0.50:
            return f"Low ({soc_pct:.2f}%) — Below 0.50% critical limit. High response to FYM, green manuring and compost."
        elif 0.50 <= soc_pct <= 0.75:
            return f"Medium ({soc_pct:.2f}%) — Moderate biological fertility; maintain with crop residue incorporation."
        else:
            return f"High ({soc_pct:.2f}%) — Desirable biological activity and good structural aggregate stability."

    @staticmethod
    def interpret_soil_temperature(temp_c: float | None) -> str:
        """Interpret root-zone soil temperature in Celsius."""
        if temp_c is None:
            return "Not Available"
        if temp_c < 12.0:
            return f"Cool ({temp_c:.1f}°C) — Slow root nutrient uptake and microbial activity."
        elif 12.0 <= temp_c <= 28.0:
            return f"Optimal ({temp_c:.1f}°C) — Favorable for root respiration and beneficial mycorrhizae."
        elif 28.0 < temp_c <= 35.0:
            return f"Warm ({temp_c:.1f}°C) — High evaporation rate; organic mulch recommended."
        else:
            return f"High Heat ({temp_c:.1f}°C) — Root thermal stress; maintain canopy cover and irrigation."
