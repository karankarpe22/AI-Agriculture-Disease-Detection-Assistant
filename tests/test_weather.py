"""Unit tests for Weather service and disease risk analysis."""
from __future__ import annotations

from src.weather.weather_service import WeatherService, get_weather_service


def test_weather_service_fetches_real_data():
    service = get_weather_service()
    data = service.get_weather("Pune")

    assert "temperature_c" in data
    assert "humidity_percentage" in data
    assert "condition" in data
    assert "risk_analysis" in data
    assert len(data["risk_analysis"]) > 10


def test_disease_risk_analysis_logic():
    service = WeatherService()
    # High humidity and cool temperature -> late blight warning
    risk_text = service._analyze_disease_risks(temperature=18.0, humidity=92.0, precipitation=1.0)
    assert "Late Blight" in risk_text or "Phytophthora" in risk_text

    # Hot and dry -> spider mite warning
    risk_dry = service._analyze_disease_risks(temperature=34.0, humidity=35.0, precipitation=0.0)
    assert "spider mite" in risk_dry.lower()
