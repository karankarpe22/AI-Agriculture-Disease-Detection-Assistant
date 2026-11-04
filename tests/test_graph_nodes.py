"""Unit tests for LangGraph node adapters in src/graph/nodes.py.

All external network calls (weather APIs, Gemini API) and heavy ML inference
(MobileNetV3 PyTorch weights, FAISS encoder) are mocked to ensure deterministic,
fast, offline execution.
Uses standard asyncio.run() to avoid dependency on external pytest-asyncio plugin.
"""
from __future__ import annotations

import asyncio
import io
import operator
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from src.graph.nodes import (
    quality_node,
    prediction_node,
    weather_node,
    soil_node,
    rag_node,
    gradcam_node,
    context_aggregation_node,
    gemini_guidance_node,
    CROP_TRANSLATIONS,
    DISEASE_TRANSLATIONS,
)
from src.graph.state import AgricultureAssistantState, create_initial_state


# =============================================================================
# 1. Quality Node Tests
# =============================================================================
def test_quality_node_valid_image():
    """Verify quality_node accepts valid leaf image and records execution stage."""
    state = create_initial_state(request_type="analyze", image_input=b"dummy_jpeg_bytes")

    mock_quality_res = {
        "is_valid": True,
        "issues": [],
        "metrics": {"width": 256, "height": 256, "brightness": 120.0, "blur_score": 150.0},
        "message": "Image quality check passed.",
    }

    with patch("src.graph.nodes.check_image_quality", return_value=mock_quality_res) as mock_check:
        result = asyncio.run(quality_node(state))

    mock_check.assert_called_once_with(b"dummy_jpeg_bytes")
    assert result["quality_result"]["is_valid"] is True
    assert result["completed_stages"] == ["quality_check"]
    assert "execution_status" not in result  # remains pending/unaltered when valid


def test_quality_node_invalid_image():
    """Verify quality_node flags blur/lighting failures and sets quality_failed status."""
    state = create_initial_state(request_type="analyze", image_input=b"blurry_bytes")

    mock_quality_res = {
        "is_valid": False,
        "issues": ["excessive_blur"],
        "metrics": {"blur_score": 12.0},
        "message": "The image appears blurry. Please hold steady.",
    }

    with patch("src.graph.nodes.check_image_quality", return_value=mock_quality_res):
        result = asyncio.run(quality_node(state))

    assert result["quality_result"]["is_valid"] is False
    assert result["execution_status"] == "quality_failed"
    assert result["completed_stages"] == ["quality_check"]
    assert "The image appears blurry" in result["error_information"][0]


def test_quality_node_missing_image():
    """Verify quality_node handles missing image input without throwing unhandled exceptions."""
    state = create_initial_state(request_type="analyze", image_input=None)
    result = asyncio.run(quality_node(state))

    assert result["quality_result"]["is_valid"] is False
    assert result["execution_status"] == "quality_failed"
    assert result["completed_stages"] == ["quality_check"]
    assert "No image provided" in result["error_information"][0]


# =============================================================================
# 2. Prediction Node Tests
# =============================================================================
def test_prediction_node_success():
    """Verify prediction_node invokes MobileNetV3 and populates bilingual crop/disease metadata."""
    state = create_initial_state(request_type="analyze", image_input=b"valid_leaf_bytes")

    mock_pred_res = {
        "success": True,
        "crop": "Tomato",
        "disease": "Late blight",
        "raw_class": "Tomato_Late_blight",
        "confidence": 0.942,
        "confidence_percentage": 94.2,
        "is_low_confidence": False,
        "top_predictions": [{"crop": "Tomato", "disease": "Late blight", "confidence": 0.942}],
    }

    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = mock_pred_res

    with patch("src.graph.nodes.get_predictor", return_value=mock_predictor):
        result = asyncio.run(prediction_node(state))

    mock_predictor.predict.assert_called_once_with(b"valid_leaf_bytes", bypass_quality_check=True)
    assert result["crop"] == "Tomato"
    assert result["crop_en"] == "Tomato"
    assert result["crop_mr"] == "टोमॅटो"
    assert result["disease"] == "Late blight"
    assert "उशिरा येणारा करपा" in result["disease_mr"]
    assert result["confidence"] == 0.942
    assert result["is_low_confidence"] is False
    assert result["completed_stages"] == ["prediction"]


def test_prediction_node_skips_when_quality_failed():
    """Verify prediction_node bypasses execution if quality check has already failed."""
    state = create_initial_state(request_type="analyze", image_input=b"bad_image")
    state["execution_status"] = "quality_failed"

    mock_predictor = MagicMock()
    with patch("src.graph.nodes.get_predictor", return_value=mock_predictor):
        result = asyncio.run(prediction_node(state))

    mock_predictor.predict.assert_not_called()
    assert result == {}


def test_prediction_node_failure_handling():
    """Verify prediction_node records execution error when inference fails."""
    state = create_initial_state(request_type="analyze", image_input=b"corrupted_bytes")

    mock_pred_res = {
        "success": False,
        "error_type": "image_decode_error",
        "warning": "Could not decode image bytes",
    }
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = mock_pred_res

    with patch("src.graph.nodes.get_predictor", return_value=mock_predictor):
        result = asyncio.run(prediction_node(state))

    assert result["execution_status"] == "error"
    assert result["completed_stages"] == ["prediction"]
    assert "Could not decode image" in result["error_information"][0]


# =============================================================================
# 3. Weather Node Tests
# =============================================================================
def test_weather_node_preserves_localization():
    """Verify weather_node fetches weather and applies Marathi localization fields."""
    state = create_initial_state(location="Pune")

    mock_weather_raw = {
        "success": True,
        "location": "Pune",
        "temperature_c": 26.5,
        "humidity_percentage": 78,
        "condition": "Overcast",
        "risk_analysis": "High humidity promotes late blight sporulation.",
    }
    mock_service = MagicMock()
    mock_service.get_weather.return_value = mock_weather_raw

    with patch("src.graph.nodes.get_weather_service", return_value=mock_service):
        result = asyncio.run(weather_node(state))

    mock_service.get_weather.assert_called_once_with(city="Pune")
    w = result["weather_result"]
    assert w["location"] == "Pune"
    assert "location_mr" in w
    assert "पुणे" in w["location_mr"]
    assert "condition_mr" in w
    assert "risk_analysis_mr" in w
    assert result["completed_stages"] == ["weather"]


def test_weather_node_failure_handling():
    """Verify weather_node falls back safely when external weather API throws exception."""
    state = create_initial_state(location="Nashik")

    mock_service = MagicMock()
    mock_service.get_weather.side_effect = RuntimeError("Connection timeout to weather API")

    with patch("src.graph.nodes.get_weather_service", return_value=mock_service):
        result = asyncio.run(weather_node(state))

    assert result["weather_result"]["success"] is False
    assert result["weather_result"]["provider"] == "Fallback"
    assert "location_mr" in result["weather_result"]
    assert result["completed_stages"] == ["weather"]
    assert "Weather service error" in result["error_information"][0]


# =============================================================================
# 4. Soil Node Tests
# =============================================================================
def test_soil_node_preserves_synthetic_disclosures():
    """Verify soil_node queries synthetic soil service and retains mandatory synthetic tags."""
    state = create_initial_state(location="Pune", crop="Tomato")

    mock_soil_raw = {
        "record_id": "SYN-SOIL-MH-PUN-001",
        "location": "Pune",
        "crop_type": "Tomato",
        "soil_ph": 7.2,
        "soil_moisture_percent": 28.0,
        "nitrogen_kg_ha": 210.0,
        "phosphorus_kg_ha": 24.0,
        "potassium_kg_ha": 310.0,
        "organic_carbon_percent": 0.65,
        "data_type": "synthetic",
        "is_synthetic": True,
        "disclaimer": "SYNTHETIC DEMONSTRATION DATA ONLY",
        "interpretation": {"ph_status": "Neutral (Optimum)"},
    }
    mock_service = MagicMock()
    mock_service.get_soil_health.return_value = mock_soil_raw

    with patch("src.graph.nodes.get_soil_service", return_value=mock_service):
        result = asyncio.run(soil_node(state))

    mock_service.get_soil_health.assert_called_once_with(location="Pune", crop="Tomato")
    s = result["soil_result"]
    assert s["data_type"] == "synthetic"
    assert s["is_synthetic"] is True
    assert "disclaimer" in s
    assert result["completed_stages"] == ["soil"]


def test_soil_node_failure_handling():
    """Verify soil_node falls back gracefully on error while retaining synthetic labeling."""
    state = create_initial_state(location="UnknownLoc", crop="Potato")

    mock_service = MagicMock()
    mock_service.get_soil_health.side_effect = RuntimeError("Dataset loading failed")

    with patch("src.graph.nodes.get_soil_service", return_value=mock_service):
        result = asyncio.run(soil_node(state))

    s = result["soil_result"]
    assert s["is_fallback"] is True
    assert s["data_type"] == "synthetic"
    assert s["is_synthetic"] is True
    assert result["completed_stages"] == ["soil"]
    assert "Soil service error" in result["error_information"][0]


# =============================================================================
# 5. RAG Node Tests
# =============================================================================
def test_rag_node_preserves_metadata_and_scores():
    """Verify rag_node retrieves chunks via vector search and generates deduplicated source summaries."""
    state = create_initial_state(
        crop="Tomato",
        disease="Late blight",
        user_question="What fungicide to spray?",
    )

    mock_chunks = [
        {
            "chunk_id": "ICAR_TOMATO_01",
            "title": "ICAR Late Blight Management",
            "source": "ICAR-IIHR",
            "heading": "Chemical Control",
            "content": "Spray Mancozeb 75% WP @ 2.5 g/L preventively.",
            "file": "icar_tomato.md",
            "similarity_score": 0.884,
            "retrieval_method": "vector",
        }
    ]
    mock_sources = [{"title": "ICAR Late Blight Management", "source": "ICAR-IIHR"}]

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = mock_chunks
    mock_rag.get_sources_summary.return_value = mock_sources

    with patch("src.graph.nodes.get_rag_service", return_value=mock_rag):
        result = asyncio.run(rag_node(state))

    mock_rag.retrieve.assert_called_once_with(
        crop="Tomato",
        disease="Late blight",
        question="What fungicide to spray?",
        top_k=3,
    )
    assert len(result["retrieved_evidence"]) == 1
    assert result["retrieved_evidence"][0]["similarity_score"] == 0.884
    assert result["retrieved_evidence"][0]["retrieval_method"] == "vector"
    assert result["source_summaries"] == mock_sources
    assert result["completed_stages"] == ["rag"]


def test_rag_node_failure_handling():
    """Verify rag_node returns empty lists and logs non-fatal error when FAISS/retrieval fails."""
    state = create_initial_state(crop="Tomato", disease="Late blight")

    mock_rag = MagicMock()
    mock_rag.retrieve.side_effect = RuntimeError("FAISS index file missing")

    with patch("src.graph.nodes.get_rag_service", return_value=mock_rag):
        result = asyncio.run(rag_node(state))

    assert result["retrieved_evidence"] == []
    assert result["source_summaries"] == []
    assert result["completed_stages"] == ["rag"]
    assert "RAG retrieval error" in result["error_information"][0]


# =============================================================================
# 6. Grad-CAM Node Tests
# =============================================================================
def test_gradcam_node_raw_base64_format():
    """Verify gradcam_node outputs raw base64 string WITHOUT data URI prefix."""
    state = create_initial_state(
        request_type="analyze",
        image_input=b"dummy_leaf_image",
        include_gradcam=True,
    )

    dummy_img = Image.new("RGB", (10, 10), color="green")

    mock_predictor = MagicMock()
    mock_predictor.generate_gradcam.return_value = dummy_img

    with patch("src.graph.nodes.get_predictor", return_value=mock_predictor):
        result = asyncio.run(gradcam_node(state))

    assert result["gradcam_result"] is not None
    # Must NOT have data:image prefix (frontend prepends it)
    assert not result["gradcam_result"].startswith("data:image")
    assert isinstance(result["gradcam_result"], str)
    assert result["gradcam_error"] is None
    assert result["completed_stages"] == ["gradcam"]


def test_gradcam_node_non_fatal_failure():
    """Verify gradcam_node handles generation exception without raising unhandled error."""
    state = create_initial_state(
        request_type="analyze",
        image_input=b"dummy_bytes",
        include_gradcam=True,
    )

    mock_predictor = MagicMock()
    mock_predictor.generate_gradcam.side_effect = RuntimeError("CUDA out of memory for backward hook")

    with patch("src.graph.nodes.get_predictor", return_value=mock_predictor):
        result = asyncio.run(gradcam_node(state))

    assert result["gradcam_result"] is None
    assert "CUDA out of memory" in result["gradcam_error"]
    assert result["completed_stages"] == ["gradcam"]
    assert "Grad-CAM error" in result["error_information"][0]


def test_gradcam_node_skipped_when_not_requested():
    """Verify gradcam_node returns empty dict when include_gradcam is False."""
    state = create_initial_state(include_gradcam=False)
    result = asyncio.run(gradcam_node(state))
    assert result == {}


# =============================================================================
# 7. Context Aggregation Node Tests
# =============================================================================
def test_context_aggregation_node():
    """Verify context_aggregation_node aggregates context, retains synthetic disclaimer, and counts evidence."""
    state = create_initial_state(
        crop="Potato",
        disease="Early blight",
        confidence=0.89,
        location="Nashik",
        user_question="How to control early blight?",
    )
    state["weather_result"] = {"temperature_c": 24.0, "humidity_percentage": 70}
    state["soil_result"] = {
        "soil_ph": 6.8,
        "is_synthetic": True,
        "disclaimer": "SYNTHETIC DEMONSTRATION DATA ONLY",
    }
    state["retrieved_evidence"] = [{"chunk_id": "ICAR_01"}, {"chunk_id": "ICAR_02"}]
    state["source_summaries"] = [{"title": "ICAR Advisory"}]

    result = asyncio.run(context_aggregation_node(state))

    ctx = result["aggregated_context"]
    assert ctx["crop"] == "Potato"
    assert ctx["disease"] == "Early blight"
    assert ctx["evidence_count"] == 2
    assert ctx["soil_is_synthetic"] is True
    assert "SYNTHETIC DEMONSTRATION" in ctx["soil_disclaimer"]
    assert result["completed_stages"] == ["context_aggregation"]


# =============================================================================
# 8. Gemini Guidance Node Tests
# =============================================================================
def test_gemini_guidance_node_single_invocation():
    """Verify gemini_guidance_node makes exactly ONE call to generate_guidance and populates outputs."""
    state = create_initial_state(
        crop="Tomato",
        disease="Late blight",
        confidence=0.92,
        user_question="What spray should I use?",
        language="english",
        location="Pune",
    )
    state["weather_result"] = {"location": "Pune", "temperature_c": 22.0}
    state["soil_result"] = {"soil_ph": 7.0, "is_synthetic": True}
    state["retrieved_evidence"] = [{"chunk_id": "ICAR_01"}]

    mock_gemini_guidance = {
        "explanation": "Symptoms indicate late blight caused by Phytophthora infestans.",
        "management_guidance": "Spray Mancozeb @ 2.5 g/L.",
        "prevention": "Prune lower infected foliage.",
        "precautions": "Wear PPE during spraying.",
        "direct_answer_en": "Apply Mancozeb 75% WP @ 2.5 g/L preventively.",
        "direct_answer_mr": "मँकोझेब ७५% WP २.५ ग्रॅम/लिटर प्रमाणे फवारा.",
        "en": {
            "direct_answer": "Apply Mancozeb 75% WP @ 2.5 g/L preventively.",
            "explanation": "Symptoms indicate late blight.",
        },
        "mr": {
            "direct_answer": "मँकोझेब ७५% WP २.५ ग्रॅम/लिटर प्रमाणे फवारा.",
            "explanation": "टोमॅटो पिकावर उशिरा येणाऱ्या करप्याची लक्षणे दिसत आहेत.",
        },
    }

    mock_gemini = MagicMock()
    mock_gemini.generate_guidance.return_value = mock_gemini_guidance

    with patch("src.graph.nodes.get_gemini_service", return_value=mock_gemini):
        result = asyncio.run(gemini_guidance_node(state))

    # Exactly ONE call made
    assert mock_gemini.generate_guidance.call_count == 1
    call_kwargs = mock_gemini.generate_guidance.call_args.kwargs
    assert call_kwargs["crop"] == "Tomato"
    assert call_kwargs["disease"] == "Late blight"
    assert call_kwargs["farmer_question"] == "What spray should I use?"

    assert result["direct_answer"] == "Apply Mancozeb 75% WP @ 2.5 g/L preventively."
    assert result["direct_answer_en"] == "Apply Mancozeb 75% WP @ 2.5 g/L preventively."
    assert result["direct_answer_mr"] == "मँकोझेब ७५% WP २.५ ग्रॅम/लिटर प्रमाणे फवारा."
    assert result["execution_status"] == "success"
    assert result["completed_stages"] == ["gemini_guidance"]


def test_gemini_guidance_node_standalone_crop_inference():
    """Verify gemini_guidance_node infers crop from Marathi text query when crop is None."""
    state = create_initial_state(
        crop=None,
        disease=None,
        user_question="टोमॅटो पिकासाठी कोणते खत योग्य आहे?",
        language="marathi",
    )

    mock_gemini = MagicMock()
    mock_gemini.generate_guidance.return_value = {
        "en": {"direct_answer": "Use NPK 10:26:26."},
        "mr": {"direct_answer": "१०:२६:२६ खत वापरा."},
        "direct_answer_en": "Use NPK 10:26:26.",
        "direct_answer_mr": "१०:२६:२६ खत वापरा.",
    }

    with patch("src.graph.nodes.get_gemini_service", return_value=mock_gemini):
        result = asyncio.run(gemini_guidance_node(state))

    # Inferred crop must be Tomato
    assert result["crop"] == "Tomato"
    assert result["crop_mr"] == "टोमॅटो"
    assert result["direct_answer"] == "१०:२६:२६ खत वापरा."


def test_gemini_guidance_node_failure_fallback():
    """Verify gemini_guidance_node falls back to grounded ICAR dictionary on API exception."""
    state = create_initial_state(
        crop="Bell pepper",
        disease="Bacterial spot",
        user_question="How to cure bacterial spot?",
    )

    mock_gemini = MagicMock()
    mock_gemini.generate_guidance.side_effect = RuntimeError("Gemini API quota exceeded (429)")

    with patch("src.graph.nodes.get_gemini_service", return_value=mock_gemini):
        result = asyncio.run(gemini_guidance_node(state))

    assert result["execution_status"] == "success"
    assert "gemini_guidance" in result
    assert result["gemini_guidance"]["powered_by"] == "ICAR Grounded Advisory"
    assert result["completed_stages"] == ["gemini_guidance"]


# =============================================================================
# 9. Parallel List Reducer Demonstration
# =============================================================================
def test_parallel_list_reducers_behavior():
    """Verify that operator.add reducer safely combines completed_stages and error_information

    Simulates LangGraph's fan-out execution where multiple concurrent nodes return updates
    simultaneously without overwriting each other.
    """
    initial_stages: list[str] = ["quality_check", "prediction"]
    initial_errors: list[str] = []

    # Node updates simulated from parallel branches:
    weather_update = {"completed_stages": ["weather"], "error_information": ["Weather fallback used"]}
    soil_update = {"completed_stages": ["soil"], "error_information": []}
    rag_update = {"completed_stages": ["rag"], "error_information": []}

    # Simulate LangGraph's operator.add reducer application
    merged_stages = initial_stages
    merged_errors = initial_errors

    for update in [weather_update, soil_update, rag_update]:
        if "completed_stages" in update:
            merged_stages = operator.add(merged_stages, update["completed_stages"])
        if "error_information" in update and update["error_information"]:
            merged_errors = operator.add(merged_errors, update["error_information"])

    # All branches are retained in order without overwrite
    assert merged_stages == ["quality_check", "prediction", "weather", "soil", "rag"]
    assert len(merged_errors) == 1
    assert "Weather fallback used" in merged_errors[0]


# =============================================================================
# 10. Localization Utility and Re-export Tests
# =============================================================================
def test_localization_utility_and_backend_reexport():
    """Verify localization utility enrichment and re-export parity in backend.main."""
    from src.utils.localization import (
        localize_weather_data as util_localize,
        CITY_TRANSLATIONS_MR,
        WEATHER_CONDITIONS_MR,
        CROP_TRANSLATIONS as UTIL_CROPS,
        DISEASE_TRANSLATIONS as UTIL_DISEASES,
    )
    import backend.main as main_mod

    # Verify re-export identity in backend.main
    assert main_mod.localize_weather_data is util_localize
    assert main_mod.CITY_TRANSLATIONS_MR is CITY_TRANSLATIONS_MR
    assert main_mod.WEATHER_CONDITIONS_MR is WEATHER_CONDITIONS_MR
    assert main_mod.CROP_TRANSLATIONS is UTIL_CROPS
    assert main_mod.DISEASE_TRANSLATIONS is UTIL_DISEASES

    # Test weather localization output
    raw_weather = {
        "location": "Pune",
        "temperature_c": 24.5,
        "humidity_percentage": 82,
        "condition": "Overcast",
        "risk_analysis": "High humidity promotes fungal spore development.",
    }
    enriched = util_localize(raw_weather)
    assert enriched["location"] == "Pune"
    assert "location_mr" in enriched
    assert "पुणे" in enriched["location_mr"]
    assert enriched["condition_mr"] == "पूर्ण ढगाळ"
    assert "आर्द्रता (82%)" in enriched["risk_analysis_mr"]


# =============================================================================
# 11. Soil Node Match vs Fallback Safety Tests
# =============================================================================
def test_soil_node_match_vs_fallback_distinction():
    """Verify soil_node clearly preserves match_method, is_fallback, and synthetic disclosures."""
    # 1. Genuine exact location match
    state_exact = create_initial_state(location="Pune", crop="Tomato")
    res_exact = asyncio.run(soil_node(state_exact))
    soil_exact = res_exact["soil_result"]

    assert soil_exact["data_type"] == "synthetic"
    assert soil_exact["is_synthetic"] is True
    assert soil_exact["is_fallback"] is False
    assert soil_exact["match_method"] == "exact_location_crop"
    assert "synthetic demonstration" in soil_exact["disclaimer"].lower()

    # 2. Unknown location fallback to statewide baseline
    state_unknown = create_initial_state(location="NonExistentTown999", crop="Tomato")
    res_unknown = asyncio.run(soil_node(state_unknown))
    soil_unknown = res_unknown["soil_result"]

    assert soil_unknown["is_fallback"] is True
    assert soil_unknown["match_method"] == "regional_baseline"
    assert soil_unknown["data_type"] == "synthetic"

    # 3. Exception fallback in node
    mock_service = MagicMock()
    mock_service.get_soil_health.side_effect = RuntimeError("Database file unreadable")
    with patch("src.graph.nodes.get_soil_service", return_value=mock_service):
        res_exc = asyncio.run(soil_node(state_exact))

    soil_exc = res_exc["soil_result"]
    assert soil_exc["is_fallback"] is True
    assert soil_exc["match_method"] == "exception_fallback"
    assert soil_exc["data_type"] == "synthetic"
    assert soil_exc["is_synthetic"] is True
    assert "synthetic demonstration" in soil_exc["disclaimer"].lower()
    assert len(res_exc["error_information"]) == 1


# =============================================================================
# 12. Clean Import and No Circular Dependency Safety Test
# =============================================================================
def test_import_safety_and_no_circular_dependency():
    """Verify that state, nodes, utils, and backend modules can be imported in any order without circular imports."""
    import src.graph.state
    import src.graph.nodes
    import src.utils.localization
    import backend.main

    assert hasattr(src.graph.state, "AgricultureAssistantState")
    assert hasattr(src.graph.nodes, "weather_node")
    assert hasattr(src.utils.localization, "localize_weather_data")
    assert hasattr(backend.main, "app")


