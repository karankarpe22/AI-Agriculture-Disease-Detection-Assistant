"""Unit tests for LangGraph typed state design (AgricultureAssistantState)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.graph.state import AgricultureAssistantState, create_initial_state, state_to_api_response
from src.weather.weather_service import get_weather_service
from src.soil.soil_service import get_soil_service
from src.rag.vector_store import get_rag_service


def test_state_import_and_instantiation():
    """Verify that AgricultureAssistantState can be imported and instantiated as a typed dict."""
    state: AgricultureAssistantState = {
        "request_type": "analyze",
        "language": "english",
        "location": "Pune",
    }
    assert state["request_type"] == "analyze"
    assert state["language"] == "english"
    assert state["location"] == "Pune"


def test_optional_fields_can_be_omitted():
    """Verify that any and all optional fields can be omitted without schema errors (total=False)."""
    minimal_state: AgricultureAssistantState = {}
    assert isinstance(minimal_state, dict)
    assert len(minimal_state) == 0

    state_with_only_type: AgricultureAssistantState = {"request_type": "ask"}
    assert state_with_only_type.get("request_type") == "ask"
    assert state_with_only_type.get("image_input") is None
    assert state_with_only_type.get("quality_result") is None


def test_image_analysis_request_representation():
    """Verify state representation for a full end-to-end leaf image analysis workflow (/analyze)."""
    mock_image_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # minimal dummy JPEG header

    state: AgricultureAssistantState = create_initial_state(
        request_type="analyze",
        user_question="What spray should I use?",
        language="english",
        location="Pune",
        image_input=mock_image_bytes,
        include_gradcam=True,
    )

    assert state["request_type"] == "analyze"
    assert state["image_input"] == mock_image_bytes
    assert state["include_gradcam"] is True
    assert state["execution_status"] == "pending"

    # Populate simulated vision pipeline outputs
    state["quality_result"] = {
        "is_valid": True,
        "issues": [],
        "metrics": {"width": 256, "height": 256, "brightness": 120.0, "blur_score": 140.0},
        "message": "Quality passed",
    }
    state["crop"] = "Tomato"
    state["disease"] = "Late blight"
    state["confidence"] = 0.94
    state["is_low_confidence"] = False
    state["prediction_result"] = {
        "success": True,
        "crop": "Tomato",
        "disease": "Late blight",
        "raw_class": "Tomato_Late_blight",
        "confidence": 0.94,
        "confidence_percentage": 94.0,
        "is_low_confidence": False,
        "top_predictions": [],
    }
    state["gradcam_result"] = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."

    assert state["quality_result"]["is_valid"] is True
    assert state["crop"] == "Tomato"
    assert state["disease"] == "Late blight"
    assert state["confidence"] == 0.94
    assert state["gradcam_result"].startswith("data:image/jpeg;base64,")


def test_text_only_qa_request_representation():
    """Verify state representation for standalone agricultural Q&A (/ask) without an image."""
    state: AgricultureAssistantState = create_initial_state(
        request_type="ask",
        user_question="Can I use organic neem oil for tomato late blight?",
        language="marathi",
        location="Nashik",
        crop="Tomato",
        disease="Late blight",
    )

    assert state["request_type"] == "ask"
    assert state["image_input"] is None
    assert state["include_gradcam"] is False
    assert state["crop"] == "Tomato"
    assert state["disease"] == "Late blight"
    assert state["language"] == "marathi"
    assert state["location"] == "Nashik"

    # Ensure vision fields are omitted
    assert "quality_result" not in state or state["quality_result"] is None
    assert "gradcam_result" not in state or state["gradcam_result"] is None


def test_error_and_execution_tracking():
    """Verify tracking of execution status, error messages, and completed pipeline stages."""
    state: AgricultureAssistantState = create_initial_state(request_type="analyze")

    # Simulate stage completion
    state["completed_stages"] = ["input_validation", "quality_check"]
    assert len(state["completed_stages"]) == 2

    # Simulate quality failure
    state["execution_status"] = "quality_failed"
    state["error_information"] = ["Image too blurry: blur variance 12.4 < 35.0 threshold"]

    assert state["execution_status"] == "quality_failed"
    assert len(state["error_information"]) == 1
    assert "too blurry" in state["error_information"][0]


def test_existing_services_data_storage_compatibility():
    """Verify that actual result dictionaries from real project services can be stored without transformation."""
    weather_service = get_weather_service()
    soil_service = get_soil_service()
    rag_service = get_rag_service()

    weather_data = weather_service.get_weather(city="Pune")
    soil_data = soil_service.get_soil_health(location="Pune", crop="Tomato")
    evidence = rag_service.retrieve(crop="Tomato", disease="Late blight", question="What fungicide to spray?", top_k=2)
    sources = rag_service.get_sources_summary(evidence)

    state: AgricultureAssistantState = create_initial_state(
        request_type="analyze",
        location="Pune",
        crop="Tomato",
        disease="Late blight",
    )

    # Store actual live results directly in state
    state["weather_result"] = weather_data
    state["soil_result"] = soil_data
    state["retrieved_evidence"] = evidence
    state["source_summaries"] = sources
    state["execution_status"] = "success"

    # Verify weather fields
    assert state["weather_result"] is not None
    assert "temperature_c" in state["weather_result"]
    assert "humidity_percentage" in state["weather_result"]
    assert "risk_analysis" in state["weather_result"]

    # Verify soil synthetic disclosures
    assert state["soil_result"] is not None
    assert state["soil_result"]["data_type"] == "synthetic"
    assert state["soil_result"]["is_synthetic"] is True
    assert "soil_ph" in state["soil_result"]
    assert "interpretation" in state["soil_result"]

    # Verify RAG FAISS evidence
    assert state["retrieved_evidence"] is not None
    assert len(state["retrieved_evidence"]) > 0
    assert state["retrieved_evidence"][0]["retrieval_method"] == "vector"
    assert "chunk_id" in state["retrieved_evidence"][0]
    assert "similarity_score" in state["retrieved_evidence"][0]

    # Verify sources
    assert state["source_summaries"] is not None
    assert len(state["source_summaries"]) > 0


def test_state_to_api_response_analyze_contract():
    """Verify state_to_api_response produces the exact JSON dictionary expected by frontend for /analyze."""
    state: AgricultureAssistantState = create_initial_state(
        request_type="analyze",
        user_question="What care should be taken?",
        language="english",
        location="Pune",
        crop="Tomato",
        disease="Late blight",
        confidence=0.92,
    )
    state["crop_en"] = "Tomato"
    state["crop_mr"] = "टोमॅटो"
    state["disease_en"] = "Late blight"
    state["disease_mr"] = "उशिरा येणारा करपा"
    state["direct_answer_en"] = "Apply Mancozeb 75% WP @ 2.5 g/L preventively."
    state["direct_answer_mr"] = "मँकोझेब ७५% WP (२.५ ग्रॅम/लिटर) फवारणी करा."
    state["execution_status"] = "success"
    state["quality_result"] = {"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}
    state["prediction_result"] = {
        "success": True,
        "raw_class": "Tomato_Late_blight",
        "confidence": 0.92,
        "confidence_percentage": 92.0,
        "is_low_confidence": False,
        "top_predictions": [],
    }

    response = state_to_api_response(state)

    # Required top-level /analyze keys
    expected_keys = [
        "success", "crop", "crop_en", "crop_mr", "disease", "disease_en", "disease_mr",
        "confidence", "confidence_percentage", "is_low_confidence", "quality", "weather",
        "soil", "evidence", "sources", "guidance", "direct_answer", "direct_answer_en",
        "direct_answer_mr", "farmer_question", "gradcam_base64",
    ]
    for key in expected_keys:
        assert key in response, f"Missing required response key: {key}"

    assert response["success"] is True
    assert response["crop"] == "Tomato"
    assert response["direct_answer"] == "Apply Mancozeb 75% WP @ 2.5 g/L preventively."


def test_state_to_api_response_ask_contract_marathi():
    """Verify state_to_api_response produces the exact JSON dictionary expected for Marathi /ask."""
    state: AgricultureAssistantState = create_initial_state(
        request_type="ask",
        user_question="हा रोग कसा रोखावा?",
        language="marathi",
        location="Pune",
        crop="Potato",
        disease="Early blight",
        confidence=0.88,
    )
    state["crop_en"] = "Potato"
    state["crop_mr"] = "बटाटा"
    state["disease_en"] = "Early blight"
    state["disease_mr"] = "लवकर येणारा करपा"
    state["direct_answer_en"] = "Crop rotation and fungicide spray."
    state["direct_answer_mr"] = "पीक फेरपालट करा आणि शिफारशीत बुरशीनाशक फवारा."
    state["execution_status"] = "success"

    response = state_to_api_response(state)

    assert response["success"] is True
    assert response["crop"] == "बटाटा"
    assert response["disease"] == "लवकर येणारा करपा"
    assert response["direct_answer"] == "पीक फेरपालट करा आणि शिफारशीत बुरशीनाशक फवारा."
    assert "question" in response
    assert response["question"] == "हा रोग कसा रोखावा?"

