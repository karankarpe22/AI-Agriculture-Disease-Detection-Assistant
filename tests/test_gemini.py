"""Unit tests for Gemini guidance service and Marathi formatting."""
from __future__ import annotations

from src.gemini.guidance_service import GeminiGuidanceService, get_gemini_service


def test_guidance_generation_structure():
    service = get_gemini_service()
    weather = {
        "location": "Pune",
        "temperature_c": 24.0,
        "humidity_percentage": 85,
        "condition": "Cloudy",
        "risk_analysis": "High humidity favors fungal leaf spots.",
    }
    evidence = [
        {
            "source": "ICAR-IIVR",
            "title": "Early Blight Advisory",
            "heading": "Recommended Chemical Control",
            "content": "Spray Mancozeb 75% WP @ 2.5 g/L.",
        }
    ]

    result = service.generate_guidance(
        crop="Tomato",
        disease="Early blight",
        confidence=0.91,
        weather=weather,
        retrieved_evidence=evidence,
        farmer_question="What spray should I use?",
        language="english",
    )

    assert "explanation" in result
    assert "management_guidance" in result
    assert "prevention" in result
    assert "precautions" in result
    assert result["language"] == "english"


def test_marathi_guidance_generation():
    service = get_gemini_service()
    weather = {
        "location": "Nashik",
        "temperature_c": 22.0,
        "humidity_percentage": 90,
        "condition": "Overcast",
        "risk_analysis": "Cool and humid.",
    }
    evidence = [
        {
            "source": "ICAR-CPRI",
            "title": "Late Blight Management",
            "heading": "Chemical Control",
            "content": "Spray Ridomil Gold @ 2.5 g/L.",
        }
    ]

    result_mr = service.generate_guidance(
        crop="Potato",
        disease="Late blight",
        confidence=0.88,
        weather=weather,
        retrieved_evidence=evidence,
        farmer_question="काय उपाय करावा?",
        language="marathi",
    )

    assert result_mr["language"] == "marathi"
    assert len(result_mr["explanation"]) > 20
    assert len(str(result_mr["management_guidance"])) > 20
