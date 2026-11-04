"""LangGraph shared state schema for the AI Agriculture Assistant.

Defines the typed dictionary schema passed across graph nodes for:
1. End-to-end leaf image disease diagnosis (/analyze)
2. Standalone agricultural question answering (/ask)
3. Dedicated CV disease classification (/predict)

All fields are fully compatible with Python 3.13.5 and Pydantic v2.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class AgricultureAssistantState(TypedDict, total=False):
    """Shared typed state dictionary for the AI Agriculture Assistant workflow.
    
    Fields are grouped into five architectural categories:
    A. Request Inputs
    B. Image Processing
    C. Context Services
    D. Final Response
    E. Execution Tracking
    """

    # =========================================================================
    # A. Request Inputs
    # =========================================================================
    request_type: str
    """Type of incoming request: 'analyze' (image+context), 'ask' (Q&A), or 'predict' (vision only)."""

    user_question: Optional[str]
    """Farmer's natural language question in English, Marathi, or Romanized script."""

    language: str
    """Target response language: 'english' or 'marathi' (defaults to 'english')."""

    location: Optional[str]
    """Farmer's locality, taluka, or district (e.g., 'Pune', 'Nashik', 'Baramati')."""

    image_input: Optional[bytes]
    """Raw byte content of the uploaded leaf image. Stored once; not duplicated across state."""

    include_gradcam: bool
    """Toggle flag requesting visual Grad-CAM saliency map generation."""

    # =========================================================================
    # B. Image Processing
    # =========================================================================
    quality_result: Optional[dict[str, Any]]
    """Output of ImageQualityChecker: is_valid, issues, metrics, and guidance message."""

    prediction_result: Optional[dict[str, Any]]
    """Raw output of DiseasePredictor: top predictions, raw class, checkpoint, etc."""

    crop: Optional[str]
    """Canonical predicted or resolved crop name (e.g., 'Tomato', 'Potato', 'Bell pepper')."""

    disease: Optional[str]
    """Canonical predicted or resolved disease name (e.g., 'Early blight', 'Late blight')."""

    confidence: float
    """Softmax probability of the top disease prediction (range 0.0 - 1.0)."""

    is_low_confidence: bool
    """Flag indicating model confidence falls below the uncertainty threshold (0.60)."""

    gradcam_result: Optional[str]
    """Base64-encoded JPEG image of the blended Grad-CAM heatmap overlay."""

    gradcam_error: Optional[str]
    """Diagnostic error message if Grad-CAM generation failed."""

    # =========================================================================
    # C. Context Services
    # =========================================================================
    weather_result: Optional[dict[str, Any]]
    """Live weather observation: temperature, humidity, condition, and pathogen risk analysis."""

    soil_result: Optional[dict[str, Any]]
    """Synthetic agro-climatic soil profile: pH, moisture %, NPK, OC, EC, and synthetic disclosures."""

    retrieved_evidence: Optional[list[dict[str, Any]]]
    """Ranked ICAR/TNAU research chunks from FAISS vector search with cosine similarity scores."""

    source_summaries: Optional[list[dict[str, str]]]
    """Deduplicated list of research source citations formatted for display."""

    # =========================================================================
    # D. Final Response
    # =========================================================================
    aggregated_context: Optional[dict[str, Any]]
    """Structured context synthesizing crop disease, weather risks, soil health, and evidence."""

    gemini_guidance: Optional[dict[str, Any]]
    """Raw structured Gemini advisory dictionary containing 'en' and 'mr' sub-objects."""

    guidance_en: Optional[dict[str, Any]]
    """Extracted English advisory block (direct_answer, explanation, management, prevention, etc.)."""

    guidance_mr: Optional[dict[str, Any]]
    """Extracted Marathi advisory block (direct_answer, explanation, management, prevention, etc.)."""

    direct_answer: Optional[str]
    """Primary actionable answer text in the user's selected language."""

    direct_answer_en: Optional[str]
    """English direct answer addressing the farmer's specific question or immediate step."""

    direct_answer_mr: Optional[str]
    """Marathi direct answer addressing the farmer's specific question or immediate step."""

    crop_en: Optional[str]
    """Localized crop name in English."""

    crop_mr: Optional[str]
    """Localized crop name in Marathi."""

    disease_en: Optional[str]
    """Localized disease name in English."""

    disease_mr: Optional[str]
    """Localized disease name in Marathi."""

    # =========================================================================
    # E. Execution Tracking
    # =========================================================================
    execution_status: str
    """Pipeline status: 'pending', 'success', 'quality_failed', or 'error'."""

    error_information: Annotated[list[str], operator.add]
    """List of non-fatal warnings or error details accumulated during graph execution."""

    completed_stages: Annotated[list[str], operator.add]
    """Audit log of pipeline stages completed during this workflow execution."""


def create_initial_state(
    request_type: str = "analyze",
    user_question: Optional[str] = None,
    language: str = "english",
    location: Optional[str] = "Pune",
    image_input: Optional[bytes] = None,
    include_gradcam: bool = False,
    crop: Optional[str] = None,
    disease: Optional[str] = None,
    confidence: float = 0.0,
    is_low_confidence: bool = False,
) -> AgricultureAssistantState:
    """Helper creating a well-typed initial state for LangGraph execution."""
    return AgricultureAssistantState(
        request_type=request_type,
        user_question=user_question,
        language=(language or "english").lower().strip(),
        location=location or "Pune",
        image_input=image_input,
        include_gradcam=include_gradcam,
        crop=crop,
        disease=disease,
        confidence=confidence,
        is_low_confidence=is_low_confidence,
        execution_status="pending",
        error_information=[],
        completed_stages=[],
    )


def state_to_api_response(state: AgricultureAssistantState) -> dict[str, Any]:
    """Format the LangGraph final state into the exact API response expected by the frontend.
    
    Guarantees 100% backward compatibility with existing /analyze and /ask endpoint contracts.
    """
    req_type = state.get("request_type", "analyze")
    is_mr = (state.get("language") or "").lower() in ["marathi", "mr", "मराठी"]

    crop_display = state.get("crop_mr") if is_mr else state.get("crop_en")
    disease_display = state.get("disease_mr") if is_mr else state.get("disease_en")

    direct_ans = state.get("direct_answer_mr") if (is_mr and state.get("direct_answer_mr")) else state.get("direct_answer_en")
    if not direct_ans:
        direct_ans = state.get("direct_answer", "")

    # Base response payload shared across endpoints
    base_response: dict[str, Any] = {
        "success": state.get("execution_status") == "success",
        "crop": crop_display or state.get("crop", ""),
        "crop_en": state.get("crop_en") or state.get("crop", ""),
        "crop_mr": state.get("crop_mr") or state.get("crop", ""),
        "disease": disease_display or state.get("disease", ""),
        "disease_en": state.get("disease_en") or state.get("disease", ""),
        "disease_mr": state.get("disease_mr") or state.get("disease", ""),
        "confidence": state.get("confidence", 0.0),
        "direct_answer": direct_ans,
        "direct_answer_en": state.get("direct_answer_en", ""),
        "direct_answer_mr": state.get("direct_answer_mr", ""),
        "weather": state.get("weather_result"),
        "soil": state.get("soil_result"),
        "evidence": state.get("retrieved_evidence", []),
        "sources": state.get("source_summaries", []),
        "guidance": state.get("gemini_guidance"),
        "guidance_en": state.get("guidance_en"),
        "guidance_mr": state.get("guidance_mr"),
    }

    if req_type == "ask":
        base_response["question"] = state.get("user_question")
        return base_response

    # /analyze specific additions
    pred = state.get("prediction_result") or {}
    base_response.update({
        "quality": state.get("quality_result"),
        "raw_class": pred.get("raw_class"),
        "confidence_percentage": pred.get("confidence_percentage", round(state.get("confidence", 0.0) * 100, 2)),
        "is_low_confidence": state.get("is_low_confidence", False),
        "warning": pred.get("warning"),
        "top_predictions": pred.get("top_predictions", []),
        "farmer_question": state.get("user_question"),
        "gradcam_base64": state.get("gradcam_result"),
    })

    # If quality check failed or pipeline error occurred, reflect error structure
    if state.get("execution_status") == "quality_failed":
        base_response["success"] = False
        base_response["crop"] = None
        base_response["disease"] = None
        base_response["error_type"] = pred.get("error_type", "quality_check_failed")
        base_response["warning"] = (state.get("quality_result") or {}).get("message")
    elif state.get("execution_status") == "error":
        base_response["success"] = False
        base_response["crop"] = None
        base_response["disease"] = None
        base_response["error_type"] = pred.get("error_type", "pipeline_error")
        err_info = state.get("error_information") or []
        base_response["warning"] = pred.get("warning") or (err_info[-1] if err_info else "Workflow execution error.")

    return base_response

