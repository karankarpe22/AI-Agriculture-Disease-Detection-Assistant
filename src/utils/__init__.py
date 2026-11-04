"""Utilities package for AI Agriculture Assistant."""
from src.utils.voice_service import VoiceService, get_voice_service
from src.utils.localization import (
    CROP_TRANSLATIONS,
    DISEASE_TRANSLATIONS,
    CITY_TRANSLATIONS_MR,
    WEATHER_CONDITIONS_MR,
    localize_weather_data,
)

__all__ = [
    "VoiceService",
    "get_voice_service",
    "CROP_TRANSLATIONS",
    "DISEASE_TRANSLATIONS",
    "CITY_TRANSLATIONS_MR",
    "WEATHER_CONDITIONS_MR",
    "localize_weather_data",
]
