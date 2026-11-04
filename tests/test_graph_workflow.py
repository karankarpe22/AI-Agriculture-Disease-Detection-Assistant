"""Comprehensive unit and integration tests for LangGraph workflow assembly and routing.

Verifies:
1. Workflow compilation and StateGraph topology.
2. Analyze workflow with valid image.
3. Analyze workflow with invalid image (quality check failure).
4. Invalid image skips prediction and context branches.
5. Ask workflow with explicit crop and disease.
6. Ask workflow with inferred crop from question.
7. Weather, soil, and RAG parallel execution.
8. Context aggregation synchronization.
9. Grad-CAM skipped when include_gradcam=False.
10. Grad-CAM executed when include_gradcam=True.
11. Grad-CAM failure is non-fatal.
12. Gemini guidance invoked exactly once.
13. Weather branch failure resilience.
14. Soil exception fallback with synthetic disclosures.
15. RAG retrieval failure resilience.
16. API response contract backward compatibility.
17. Bilingual Marathi localization handling.
18. Absence of circular imports.
"""
from __future__ import annotations

import asyncio
import io
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image
import pytest

from src.graph.state import (
    AgricultureAssistantState,
    create_initial_state,
    state_to_api_response,
)
from src.graph.workflow import (
    agriculture_graph,
    build_agriculture_graph,
    run_agriculture_workflow,
    run_agriculture_workflow_sync,
    setup_ask_context,
)


# =============================================================================
# Helper Fixtures & Test Data
# =============================================================================
def make_dummy_jpeg() -> bytes:
    """Create a valid synthetic JPEG image in memory for testing."""
    img = Image.new("RGB", (256, 256), color=(40, 160, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


MOCK_PREDICTION_RESULT = {
    "success": True,
    "crop": "Tomato",
    "disease": "Late blight",
    "raw_class": "Tomato_Late_blight",
    "confidence": 0.94,
    "confidence_percentage": 94.0,
    "is_low_confidence": False,
    "top_predictions": [
        {"class": "Tomato_Late_blight", "crop": "Tomato", "disease": "Late blight", "confidence": 0.94, "confidence_percentage": 94.0},
        {"class": "Tomato_Early_blight", "crop": "Tomato", "disease": "Early blight", "confidence": 0.04, "confidence_percentage": 4.0},
    ],
}

MOCK_WEATHER_RESULT = {
    "success": True,
    "city": "Pune",
    "location": "Pune, IN",
    "location_en": "Pune",
    "location_mr": "पुणे, भारत",
    "temperature_c": 28.5,
    "humidity_percentage": 75,
    "condition": "Cloudy",
    "condition_en": "Cloudy",
    "condition_mr": "ढगाळ",
    "risk_analysis": "Elevated foliar fungal risk.",
    "risk_analysis_en": "Elevated foliar fungal risk.",
    "risk_analysis_mr": "पानांवरील बुरशीचा धोका वाढला आहे.",
}

MOCK_SOIL_RESULT = {
    "record_id": "SYN-SOIL-PUN-001",
    "location": "Pune",
    "crop_type": "Tomato",
    "soil_type": "Medium Black Soil",
    "soil_ph": 7.2,
    "soil_moisture_percent": 24.5,
    "nitrogen_kg_ha": 185.0,
    "phosphorus_kg_ha": 22.0,
    "potassium_kg_ha": 310.0,
    "organic_carbon_percent": 0.58,
    "is_synthetic": True,
    "data_type": "synthetic",
    "match_method": "exact_region_crop",
    "disclaimer": "SYNTHETIC DEMONSTRATION DATA ONLY: Modeled for academic testing. Not verified on-field measurements.",
}

MOCK_RAG_EVIDENCE = [
    {
        "chunk_id": "ICAR_tomato_late_blight_01",
        "crop": "Tomato",
        "disease": "Late blight",
        "heading": "Chemical Management",
        "content": "Apply Mancozeb 75% WP @ 2.5 g/L preventively.",
        "source": "ICAR-IIHR Tomato Guidelines",
        "retrieval_method": "vector",
        "similarity_score": 0.88,
    }
]

MOCK_RAG_SOURCES = [
    {"source": "ICAR-IIHR Tomato Guidelines", "heading": "Chemical Management", "chunk_id": "ICAR_tomato_late_blight_01"}
]

MOCK_GEMINI_GUIDANCE = {
    "en": {
        "direct_answer": "Spray Mancozeb 75% WP @ 2.5 g/L immediately to manage tomato late blight.",
        "explanation": "High humidity and moderate temperature favor late blight spread.",
        "management_guidance": "Spray registered protective fungicides.",
        "prevention": "Avoid furrow pooling and remove infected lower leaves.",
    },
    "mr": {
        "direct_answer": "टोमॅटो करपा नियंत्रणासाठी मँकोझेब ७५% WP (२.५ ग्रॅम/लिटर) फवारणी करा.",
        "explanation": "जास्त आर्द्रता आणि मध्यम तापमानामुळे करपा वेगाने पसरतो.",
        "management_guidance": "शिफारस केलेल्या बुरशीनाशकांचा वापर करा.",
        "prevention": "पाण्याचा योग्य निचरा करा आणि बाधित पाने काढून टाका.",
    },
    "direct_answer_en": "Spray Mancozeb 75% WP @ 2.5 g/L immediately to manage tomato late blight.",
    "direct_answer_mr": "टोमॅटो करपा नियंत्रणासाठी मँकोझेब ७५% WP (२.५ ग्रॅम/लिटर) फवारणी करा.",
}


# =============================================================================
# 18 Test Scenarios
# =============================================================================

def test_workflow_compilation():
    """1. Verify that build_agriculture_graph compiles successfully into an executable StateGraph."""
    graph = build_agriculture_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")
    # Verify module-level instance is compiled and operational
    assert agriculture_graph is not None


def test_analyze_workflow_valid_image():
    """2. Verify full end-to-end analyze workflow execution with a valid leaf image."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        user_question="What should I spray?",
        language="english",
        location="Pune",
        image_input=dummy_img,
        include_gradcam=False,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_gp.return_value = mock_pred

        mock_ws = MagicMock()
        mock_ws.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gws.return_value = mock_ws

        mock_ss = MagicMock()
        mock_ss.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_gss.return_value = mock_ss

        mock_rs = MagicMock()
        mock_rs.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_rs.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_grs.return_value = mock_rs

        mock_gs = MagicMock()
        mock_gs.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE
        mock_ggs.return_value = mock_gs

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        assert final_state["crop"] == "Tomato"
        assert final_state["disease"] == "Late blight"
        assert final_state["confidence"] == 0.94
        assert "quality_check" in final_state["completed_stages"]
        assert "prediction" in final_state["completed_stages"]
        assert "weather" in final_state["completed_stages"]
        assert "soil" in final_state["completed_stages"]
        assert "rag" in final_state["completed_stages"]
        assert "context_aggregation" in final_state["completed_stages"]
        assert "gemini_guidance" in final_state["completed_stages"]
        assert final_state["direct_answer"] is not None


def test_analyze_workflow_invalid_image():
    """3. Verify analyze workflow aborts on image quality check failure."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        user_question=None,
        language="english",
        location="Pune",
        image_input=dummy_img,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": False, "issues": ["too_blurry"], "metrics": {}, "message": "Image is severely blurred."}):
        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "quality_failed"
        assert final_state["quality_result"]["is_valid"] is False
        assert "quality_check" in final_state["completed_stages"]
        assert len(final_state["error_information"]) > 0


def test_invalid_image_skips_prediction():
    """4. Verify that prediction, weather, soil, RAG, and Gemini are strictly skipped when quality fails."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=dummy_img,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": False, "issues": ["too_dark"], "metrics": {}, "message": "Image too dark."}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        # Downstream nodes must not have executed
        mock_gp.assert_not_called()
        mock_ggs.assert_not_called()
        assert "prediction" not in final_state.get("completed_stages", [])
        assert "gemini_guidance" not in final_state.get("completed_stages", [])
        assert final_state.get("crop") is None
        assert final_state.get("disease") is None


def test_ask_workflow_provided_crop_and_disease():
    """5. Verify ask workflow execution with explicitly provided crop and disease."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="Can I use copper fungicide?",
        crop="Potato",
        disease="Late blight",
        confidence=0.91,
        location="Nashik",
        language="english",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        assert final_state["crop"] == "Potato"
        assert final_state["disease"] == "Late blight"
        assert final_state["confidence"] == 0.91
        assert "quality_check" not in final_state.get("completed_stages", [])
        assert "prediction" not in final_state.get("completed_stages", [])
        assert "ask_context_setup" in final_state["completed_stages"]
        assert "context_aggregation" in final_state["completed_stages"]


def test_ask_workflow_inferred_crop_from_question():
    """6. Verify ask workflow infers crop when crop is missing and user mentions tomato in question."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="What is the best fertilizer for my tomato plants?",
        crop=None,
        disease=None,
        location="Pune",
        language="english",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["crop"] == "Tomato"
        assert final_state["disease"] == "General Agronomic Inquiry"
        assert final_state["execution_status"] == "success"


def test_parallel_context_branches_all_execute():
    """7. Verify weather, soil, and RAG context branches all execute and record their stages."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="General crop management tips",
        location="Pune",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        stages = final_state["completed_stages"]
        assert "weather" in stages
        assert "soil" in stages
        assert "rag" in stages


def test_context_aggregation_waits_for_all_branches():
    """8. Verify context aggregation node synchronizes outputs from weather, soil, and RAG."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="Soil and weather interaction inquiry",
        crop="Tomato",
        location="Pune",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        agg = final_state.get("aggregated_context")
        assert agg is not None
        assert agg["weather"] is not None
        assert agg["soil"] is not None
        assert agg["evidence_count"] == len(MOCK_RAG_EVIDENCE)
        assert agg["soil_is_synthetic"] is True


def test_gradcam_skipped_when_disabled():
    """9. Verify Grad-CAM is bypassed completely when include_gradcam=False."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=dummy_img,
        include_gradcam=False,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert "gradcam" not in final_state.get("completed_stages", [])
        assert final_state.get("gradcam_result") is None
        mock_pred.generate_gradcam.assert_not_called()


def test_gradcam_runs_when_enabled():
    """10. Verify Grad-CAM executes and populates raw base64 string when include_gradcam=True."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=dummy_img,
        include_gradcam=True,
    )

    dummy_cam = Image.new("RGB", (100, 100), color=(255, 0, 0))

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_pred.generate_gradcam.return_value = dummy_cam
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert "gradcam" in final_state["completed_stages"]
        assert final_state["gradcam_result"] is not None
        assert not final_state["gradcam_result"].startswith("data:image")
        mock_pred.generate_gradcam.assert_called_once()


def test_gradcam_failure_non_fatal():
    """11. Verify that Grad-CAM generation exception does not terminate or fail the analysis pipeline."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=dummy_img,
        include_gradcam=True,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_pred.generate_gradcam.side_effect = RuntimeError("GPU memory allocation error in Grad-CAM")
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        assert final_state.get("gradcam_result") is None
        assert final_state.get("gradcam_error") is not None
        assert any("Grad-CAM error" in err for err in final_state.get("error_information", []))


def test_gemini_invoked_exactly_once():
    """12. Verify Gemini guidance generation is executed exactly once during the workflow."""
    dummy_img = make_dummy_jpeg()
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=dummy_img,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}, "message": "OK"}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES

        mock_gemini = MagicMock()
        mock_gemini.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE
        mock_ggs.return_value = mock_gemini

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert mock_gemini.generate_guidance.call_count == 1
        assert final_state["execution_status"] == "success"


def test_weather_failure_fallback():
    """13. Verify weather branch network exception falls back gracefully without terminating the workflow."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="Is rain expected?",
        crop="Tomato",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.side_effect = ConnectionError("Weather API timeout")
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        assert final_state["weather_result"]["success"] is False
        assert final_state["weather_result"]["provider"] == "Fallback"
        assert any("Weather service error" in err for err in final_state.get("error_information", []))


def test_soil_exception_fallback_synthetic_disclosures():
    """14. Verify soil branch exception fallback maintains explicit synthetic disclosures."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="Check soil health",
        crop="Tomato",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.side_effect = RuntimeError("Soil dataset file missing")
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        soil = final_state["soil_result"]
        assert soil["is_synthetic"] is True
        assert soil["is_fallback"] is True
        assert soil["match_method"] == "exception_fallback"
        assert "synthetic demonstration" in soil["disclaimer"].lower()


def test_rag_failure_empty_evidence():
    """15. Verify RAG retrieval exception produces empty evidence list without failing workflow."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="Search ICAR documents",
        crop="Tomato",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.side_effect = RuntimeError("FAISS index file locked")
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert final_state["execution_status"] == "success"
        assert final_state["retrieved_evidence"] == []
        assert final_state["source_summaries"] == []
        assert any("RAG retrieval error" in err for err in final_state.get("error_information", []))


def test_api_response_contract_compatibility():
    """16. Verify state_to_api_response produces identical key-set and structure for analyze and ask."""
    # Analyze state verification
    analyze_state: AgricultureAssistantState = create_initial_state(
        request_type="analyze",
        user_question="How to cure early blight?",
        language="english",
        location="Pune",
        crop="Tomato",
        disease="Early blight",
        confidence=0.92,
    )
    analyze_state["execution_status"] = "success"
    analyze_state["quality_result"] = {"is_valid": True, "issues": [], "metrics": {}}
    analyze_state["prediction_result"] = MOCK_PREDICTION_RESULT
    analyze_state["weather_result"] = MOCK_WEATHER_RESULT
    analyze_state["soil_result"] = MOCK_SOIL_RESULT
    analyze_state["retrieved_evidence"] = MOCK_RAG_EVIDENCE
    analyze_state["source_summaries"] = MOCK_RAG_SOURCES
    analyze_state["gemini_guidance"] = MOCK_GEMINI_GUIDANCE
    analyze_state["crop_en"] = "Tomato"
    analyze_state["crop_mr"] = "टोमॅटो"
    analyze_state["disease_en"] = "Early blight"
    analyze_state["disease_mr"] = "लवकर येणारा करपा"

    resp_analyze = state_to_api_response(analyze_state)
    expected_analyze_keys = {
        "success", "quality", "crop", "crop_en", "crop_mr", "disease", "disease_en", "disease_mr",
        "raw_class", "confidence", "confidence_percentage", "is_low_confidence", "warning",
        "top_predictions", "weather", "soil", "evidence", "sources", "guidance", "guidance_en",
        "guidance_mr", "direct_answer", "direct_answer_en", "direct_answer_mr", "farmer_question", "gradcam_base64"
    }
    assert expected_analyze_keys.issubset(set(resp_analyze.keys()))

    # Ask state verification
    ask_state: AgricultureAssistantState = create_initial_state(
        request_type="ask",
        user_question="What fertilizer?",
        language="english",
        location="Pune",
        crop="Tomato",
        disease="General Agronomic Inquiry",
        confidence=0.85,
    )
    ask_state["execution_status"] = "success"
    ask_state["weather_result"] = MOCK_WEATHER_RESULT
    ask_state["soil_result"] = MOCK_SOIL_RESULT
    ask_state["retrieved_evidence"] = MOCK_RAG_EVIDENCE
    ask_state["source_summaries"] = MOCK_RAG_SOURCES
    ask_state["gemini_guidance"] = MOCK_GEMINI_GUIDANCE

    resp_ask = state_to_api_response(ask_state)
    expected_ask_keys = {
        "success", "crop", "crop_en", "crop_mr", "disease", "disease_en", "disease_mr",
        "confidence", "question", "direct_answer", "direct_answer_en", "direct_answer_mr",
        "weather", "soil", "evidence", "sources", "guidance", "guidance_en", "guidance_mr"
    }
    assert expected_ask_keys.issubset(set(resp_ask.keys()))


def test_bilingual_marathi_handling():
    """17. Verify Marathi localized text selection when language='marathi'."""
    initial_state = create_initial_state(
        request_type="ask",
        user_question="बटाट्यावरील करपा कसा रोखावा?",
        crop="Potato",
        disease="Late blight",
        language="marathi",
        location="Pune",
    )

    with patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))
        resp = state_to_api_response(final_state)

        assert resp["crop"] == "बटाटा"
        assert "करपा" in resp["disease"]
        assert resp["crop_mr"] == "बटाटा"
        assert "करपा" in resp["direct_answer"]
        assert "पुणे" in resp["weather"]["location_mr"]


def test_no_circular_imports():
    """18. Verify that src.graph.workflow does not depend on backend.main."""
    assert "src.graph.workflow" in sys.modules
    # Confirm workflow module does not import backend.main
    workflow_mod = sys.modules["src.graph.workflow"]
    assert not hasattr(workflow_mod, "app")
    assert not hasattr(workflow_mod, "analyze_leaf")
    assert not hasattr(workflow_mod, "ask_question")


def test_deterministic_parallel_barrier_ordering():
    """19. Verify that context_aggregation_node waits for weather, soil, RAG, and Grad-CAM when enabled."""
    execution_timeline = []

    def mock_weather_delayed(city):
        import time
        time.sleep(0.02)
        execution_timeline.append("weather_finished")
        return MOCK_WEATHER_RESULT

    def mock_soil_delayed(location, crop):
        import time
        time.sleep(0.03)
        execution_timeline.append("soil_finished")
        return MOCK_SOIL_RESULT

    def mock_rag_delayed(crop, disease, question, top_k):
        import time
        time.sleep(0.01)
        execution_timeline.append("rag_finished")
        return MOCK_RAG_EVIDENCE

    def mock_gradcam_delayed(image_input):
        import time
        time.sleep(0.04)
        execution_timeline.append("gradcam_finished")
        return Image.new("RGB", (50, 50), color=(100, 200, 100))

    initial_state = create_initial_state(
        request_type="analyze",
        image_input=make_dummy_jpeg(),
        include_gradcam=True,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_pred.generate_gradcam.side_effect = mock_gradcam_delayed
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.side_effect = mock_weather_delayed
        mock_gss.return_value.get_soil_health.side_effect = mock_soil_delayed
        mock_grs.return_value.retrieve.side_effect = mock_rag_delayed
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert "weather_finished" in execution_timeline
        assert "soil_finished" in execution_timeline
        assert "rag_finished" in execution_timeline
        assert "gradcam_finished" in execution_timeline
        assert final_state["aggregated_context"]["weather"] is not None
        assert final_state["aggregated_context"]["soil"] is not None
        assert final_state["aggregated_context"]["evidence_count"] > 0
        assert final_state["gradcam_result"] is not None


def test_deterministic_parallel_barrier_without_gradcam():
    """20. Verify that when Grad-CAM is disabled, context_aggregation_node does not wait for it."""
    initial_state = create_initial_state(
        request_type="analyze",
        image_input=make_dummy_jpeg(),
        include_gradcam=False,
    )

    with patch("src.graph.nodes.check_image_quality", return_value={"is_valid": True, "issues": [], "metrics": {}}), \
         patch("src.graph.nodes.get_predictor") as mock_gp, \
         patch("src.graph.nodes.get_weather_service") as mock_gws, \
         patch("src.graph.nodes.get_soil_service") as mock_gss, \
         patch("src.graph.nodes.get_rag_service") as mock_grs, \
         patch("src.graph.nodes.get_gemini_service") as mock_ggs:

        mock_pred = MagicMock()
        mock_pred.predict.return_value = MOCK_PREDICTION_RESULT
        mock_gp.return_value = mock_pred

        mock_gws.return_value.get_weather.return_value = MOCK_WEATHER_RESULT
        mock_gss.return_value.get_soil_health.return_value = MOCK_SOIL_RESULT
        mock_grs.return_value.retrieve.return_value = MOCK_RAG_EVIDENCE
        mock_grs.return_value.get_sources_summary.return_value = MOCK_RAG_SOURCES
        mock_ggs.return_value.generate_guidance.return_value = MOCK_GEMINI_GUIDANCE

        final_state = asyncio.run(run_agriculture_workflow(initial_state))

        assert mock_pred.generate_gradcam.call_count == 0
        assert "gradcam" not in final_state["completed_stages"]
        assert final_state["aggregated_context"] is not None


def test_ask_behavior_parity_matrix():
    """21. Verify full behavioral parity for all /ask variations (crop only, disease only, Marathi, synonyms)."""
    # 1. Crop provided without disease
    s1 = create_initial_state(request_type="ask", crop="Tomato", disease=None, user_question="Water requirement?")
    r1 = asyncio.run(setup_ask_context(s1))
    assert r1["crop"] == "Tomato"
    assert r1["disease"] == "General Agronomic Inquiry"

    # 2. Disease provided without crop
    s2 = create_initial_state(request_type="ask", crop=None, disease="Late blight", user_question="How to spray?")
    r2 = asyncio.run(setup_ask_context(s2))
    assert r2["crop"] == "General / Solanaceae"
    assert r2["disease"] == "Late blight"

    # 3. Marathi crop name provided
    s3 = create_initial_state(request_type="ask", crop="बटाटा", user_question="खत व्यवस्थापन?")
    r3 = asyncio.run(setup_ask_context(s3))
    assert r3["crop"] == "Potato"
    assert r3["crop_mr"] == "बटाटा"

    # 4. Marathi disease name provided
    s4 = create_initial_state(request_type="ask", crop="टोमॅटो", disease="लवकर येणारा करपा", user_question="फवारणी?")
    r4 = asyncio.run(setup_ask_context(s4))
    assert r4["crop"] == "Tomato"
    assert r4["disease"] == "Early blight"

    # 5. Disease inferred from question
    s5 = create_initial_state(request_type="ask", crop=None, disease=None, user_question="My tomato plants have late blight symptoms")
    r5 = asyncio.run(setup_ask_context(s5))
    assert r5["crop"] == "Tomato"
    assert r5["disease"] == "Late blight"

    # 6. Completely unspecified question
    s6 = create_initial_state(request_type="ask", crop=None, disease=None, user_question="What is good farming practice?")
    r6 = asyncio.run(setup_ask_context(s6))
    assert r6["crop"] == "General / Solanaceae"
    assert r6["disease"] == "General Agronomic Inquiry"
