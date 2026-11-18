"""LangGraph node adapter functions for the AI Agriculture Assistant.

Wraps existing agriculture services (vision quality checker, MobileNetV3 disease predictor,
weather microclimate service, synthetic soil service, FAISS RAG, and Gemini guidance) into
reusable, asynchronous LangGraph node functions.

All blocking operations are executed off the event loop via asyncio.to_thread().
Each node returns only its own stage additions and non-fatal errors to ensure safe
fan-out parallel execution with LangGraph operator.add reducers.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import Any, Optional

from src.gemini.guidance_service import get_gemini_service
from src.graph.state import AgricultureAssistantState
from src.prediction.service import get_predictor
from src.preprocessing.quality_check import check_image_quality
from src.rag.vector_store import get_rag_service
from src.soil.soil_service import get_soil_service
from src.utils.localization import (
    CROP_TRANSLATIONS,
    DISEASE_TRANSLATIONS,
    localize_weather_data,
)
from src.utils.memory import log_memory
from src.weather.weather_service import get_weather_service

logger = logging.getLogger("src.graph.nodes")


# =============================================================================
# A. Quality Node
# =============================================================================
async def quality_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Inspect leaf image usability (blur, lighting, resolution) using ImageQualityChecker."""
    image_input = state.get("image_input")
    if not image_input:
        error_msg = "No image provided for quality assessment."
        return {
            "quality_result": {
                "is_valid": False,
                "issues": ["missing_image"],
                "metrics": {},
                "message": error_msg,
            },
            "execution_status": "quality_failed",
            "completed_stages": ["quality_check"],
            "error_information": [error_msg],
        }

    try:
        quality_res = await asyncio.to_thread(check_image_quality, image_input)
    except Exception as e:
        error_msg = f"Quality check exception: {str(e)}"
        logger.warning(error_msg)
        return {
            "quality_result": {
                "is_valid": False,
                "issues": ["quality_checker_error"],
                "metrics": {},
                "message": "Quality check could not process image file.",
            },
            "execution_status": "quality_failed",
            "completed_stages": ["quality_check"],
            "error_information": [error_msg],
        }

    if not quality_res.get("is_valid"):
        return {
            "quality_result": quality_res,
            "execution_status": "quality_failed",
            "completed_stages": ["quality_check"],
            "error_information": [quality_res.get("message", "Image quality check failed.")],
        }

    return {
        "quality_result": quality_res,
        "completed_stages": ["quality_check"],
    }


# =============================================================================
# B. Prediction Node
# =============================================================================
async def prediction_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Execute MobileNetV3 disease inference on a quality-verified leaf image."""
    # If image quality check already failed, bypass prediction
    if state.get("execution_status") == "quality_failed":
        return {}

    image_input = state.get("image_input")
    if not image_input:
        return {
            "execution_status": "error",
            "completed_stages": ["prediction"],
            "error_information": ["Missing image input for prediction."],
        }

    try:
        predictor = get_predictor()
        # Quality check was already executed in quality_node; bypass duplicate run
        pred_res = await asyncio.to_thread(predictor.predict, image_input, bypass_quality_check=True)
        log_memory("After prediction_node inference", logger)
    except Exception as e:
        error_msg = f"Prediction model execution failed: {str(e)}"
        logger.error(error_msg)
        return {
            "execution_status": "error",
            "completed_stages": ["prediction"],
            "error_information": [error_msg],
        }

    if not pred_res.get("success"):
        return {
            "prediction_result": pred_res,
            "execution_status": "error",
            "completed_stages": ["prediction"],
            "error_information": [pred_res.get("warning") or pred_res.get("error_type", "Prediction failed.")],
        }

    crop = pred_res.get("crop")
    disease = pred_res.get("disease")
    confidence = float(pred_res.get("confidence", 0.0))
    is_low_conf = bool(pred_res.get("is_low_confidence", False))

    crop_en = CROP_TRANSLATIONS.get(crop, {}).get("en", crop or "")
    crop_mr = CROP_TRANSLATIONS.get(crop, {}).get("mr", crop or "")
    disease_en = DISEASE_TRANSLATIONS.get(disease, {}).get("en", disease or "")
    disease_mr = DISEASE_TRANSLATIONS.get(disease, {}).get("mr", disease or "")

    return {
        "prediction_result": pred_res,
        "crop": crop,
        "crop_en": crop_en,
        "crop_mr": crop_mr,
        "disease": disease,
        "disease_en": disease_en,
        "disease_mr": disease_mr,
        "confidence": confidence,
        "is_low_confidence": is_low_conf,
        "completed_stages": ["prediction"],
    }


# =============================================================================
# C. Weather Node
# =============================================================================
async def weather_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Retrieve live weather and microclimate disease risks with Marathi localization."""
    location = state.get("location") or "Pune"

    try:
        weather_service = get_weather_service()
        weather_data = await asyncio.to_thread(weather_service.get_weather, city=location)
        localized_weather = localize_weather_data(weather_data)
        return {
            "weather_result": localized_weather,
            "completed_stages": ["weather"],
        }
    except Exception as e:
        error_msg = f"Weather service error: {str(e)}"
        logger.warning(error_msg)
        fallback_weather = {
            "success": False,
            "provider": "Fallback",
            "temperature_c": 25.0,
            "humidity_percentage": 60,
            "condition": "Normal",
            "risk_analysis": "Weather service currently unavailable; follow standard preventative measures.",
        }
        localized_fallback = localize_weather_data(fallback_weather)
        return {
            "weather_result": localized_fallback,
            "completed_stages": ["weather"],
            "error_information": [error_msg],
        }


# =============================================================================
# D. Soil Node
# =============================================================================
async def soil_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Retrieve synthetic agro-climatic soil health profile and tripartite interaction."""
    location = state.get("location") or "Pune"
    crop = state.get("crop")

    try:
        soil_service = get_soil_service()
        soil_data = await asyncio.to_thread(
            soil_service.get_soil_health,
            location=location,
            crop=crop,
        )
        return {
            "soil_result": soil_data,
            "completed_stages": ["soil"],
        }
    except Exception as e:
        error_msg = f"Soil service error: {str(e)}"
        logger.warning(error_msg)
        fallback_soil = {
            "record_id": "SYN-SOIL-FALLBACK",
            "soil_type": "Medium Black Soil",
            "soil_ph": 7.0,
            "soil_moisture_percent": 25.0,
            "nitrogen_kg_ha": 180.0,
            "phosphorus_kg_ha": 22.0,
            "potassium_kg_ha": 300.0,
            "organic_carbon_percent": 0.55,
            "data_type": "synthetic",
            "is_synthetic": True,
            "is_fallback": True,
            "match_method": "exception_fallback",
            "queried_location": location,
            "queried_crop": crop or "",
            "demonstration_purpose": "Simulated regional benchmark provided because soil service raised an exception.",
            "disclaimer": "SYNTHETIC DEMONSTRATION DATA ONLY: Modeled for academic testing. Not verified on-field measurements.",
        }
        return {
            "soil_result": fallback_soil,
            "completed_stages": ["soil"],
            "error_information": [error_msg],
        }


# =============================================================================
# E. RAG Node
# =============================================================================
async def rag_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Retrieve grounded ICAR/TNAU research chunks via FAISS vector similarity search."""
    crop = state.get("crop")
    disease = state.get("disease")
    question = state.get("user_question")

    # Filter out catch-all solanaceae marker so vector search targets specific crops
    crop_filter = crop if crop and crop != "General / Solanaceae" else None
    disease_filter = disease if disease and disease != "General Agronomic Inquiry" else None

    try:
        rag_service = get_rag_service()
        evidence = await asyncio.to_thread(
            rag_service.retrieve,
            crop=crop_filter,
            disease=disease_filter,
            question=question,
            top_k=3,
        )
        sources = await asyncio.to_thread(rag_service.get_sources_summary, evidence)
        return {
            "retrieved_evidence": evidence,
            "source_summaries": sources,
            "completed_stages": ["rag"],
        }
    except Exception as e:
        error_msg = f"RAG retrieval error: {str(e)}"
        logger.warning(error_msg)
        return {
            "retrieved_evidence": [],
            "source_summaries": [],
            "completed_stages": ["rag"],
            "error_information": [error_msg],
        }


# =============================================================================
# F. Grad-CAM Node
# =============================================================================
async def gradcam_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Generate Grad-CAM visual attention heatmap as raw base64 JPEG without URI prefix."""
    # Only run when specifically requested and when quality check passed
    if not state.get("include_gradcam") or state.get("execution_status") == "quality_failed":
        return {}

    image_input = state.get("image_input")
    if not image_input:
        return {
            "gradcam_result": None,
            "gradcam_error": "No image input available for Grad-CAM.",
            "completed_stages": ["gradcam"],
            "error_information": ["No image input available for Grad-CAM."],
        }

    try:
        predictor = get_predictor()
        cam_img = await asyncio.to_thread(predictor.generate_gradcam, image_input)
        log_memory("After gradcam_node generation", logger)
        if cam_img:
            buf = io.BytesIO()
            cam_img.save(buf, format="JPEG")
            # Store strictly raw base64 string without 'data:image/jpeg;base64,' prefix
            raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return {
                "gradcam_result": raw_b64,
                "gradcam_error": None,
                "completed_stages": ["gradcam"],
            }
        else:
            return {
                "gradcam_result": None,
                "gradcam_error": "Grad-CAM generation returned empty image.",
                "completed_stages": ["gradcam"],
                "error_information": ["Grad-CAM generation returned empty image."],
            }
    except Exception as e:
        error_msg = f"Grad-CAM error: {str(e)}"
        logger.warning(error_msg)
        return {
            "gradcam_result": None,
            "gradcam_error": str(e),
            "completed_stages": ["gradcam"],
            "error_information": [error_msg],
        }


# =============================================================================
# G. Context Aggregation Node
# =============================================================================
async def context_aggregation_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Combine vision diagnosis, weather risks, synthetic soil, and research evidence."""
    soil_res = state.get("soil_result") or {}
    evidence = state.get("retrieved_evidence") or []
    sources = state.get("source_summaries") or []

    aggregated_context: dict[str, Any] = {
        "crop": state.get("crop"),
        "disease": state.get("disease"),
        "confidence": state.get("confidence", 0.0),
        "is_low_confidence": state.get("is_low_confidence", False),
        "location": state.get("location"),
        "language": state.get("language", "english"),
        "farmer_question": state.get("user_question"),
        "weather": state.get("weather_result"),
        "soil": soil_res,
        "soil_is_synthetic": bool(soil_res.get("is_synthetic", True)),
        "soil_disclaimer": soil_res.get(
            "disclaimer",
            "SYNTHETIC DEMONSTRATION DATA ONLY: Modeled for academic testing. Not verified on-field measurements.",
        ),
        "evidence_count": len(evidence),
        "evidence_chunks": evidence,
        "sources": sources,
    }

    return {
        "aggregated_context": aggregated_context,
        "completed_stages": ["context_aggregation"],
    }


# =============================================================================
# H. Gemini Guidance Node
# =============================================================================
async def gemini_guidance_node(state: AgricultureAssistantState) -> dict[str, Any]:
    """Generate evidence-grounded agronomic advisory via Gemini with English/Marathi outputs."""
    # If image quality failed, do not generate guidance
    if state.get("execution_status") == "quality_failed":
        return {}

    # Resolve crop & disease context (supporting standalone Q&A without image)
    crop = state.get("crop")
    q = state.get("user_question") or ""
    if not crop and q:
        q_lower = q.lower()
        if "tomato" in q_lower or "टोमॅटो" in q_lower:
            crop = "Tomato"
        elif "potato" in q_lower or "बटाटा" in q_lower:
            crop = "Potato"
        elif any(k in q_lower for k in ["pepper", "capsicum", "शिमला", "मिरची"]):
            crop = "Bell pepper"

    resolved_crop = crop or "General / Solanaceae"
    resolved_disease = state.get("disease") or "General Agronomic Inquiry"

    crop_en = CROP_TRANSLATIONS.get(resolved_crop, {}).get("en", resolved_crop)
    crop_mr = CROP_TRANSLATIONS.get(resolved_crop, {}).get("mr", resolved_crop)
    disease_en = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("en", resolved_disease)
    disease_mr = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("mr", resolved_disease)

    language = state.get("language") or "english"
    is_mr = language.lower() in ["marathi", "mr", "मराठी"]

    try:
        gemini = get_gemini_service()
        guidance = await asyncio.to_thread(
            gemini.generate_guidance,
            crop=resolved_crop,
            disease=resolved_disease,
            confidence=state.get("confidence", 0.85),
            weather=state.get("weather_result") or {},
            retrieved_evidence=state.get("retrieved_evidence") or [],
            farmer_question=state.get("user_question"),
            language=language,
            is_low_confidence=state.get("is_low_confidence", False),
            soil=state.get("soil_result"),
        )
    except Exception as e:
        error_msg = f"Gemini guidance service exception: {str(e)}"
        logger.warning(error_msg)
        guidance = {
            "explanation": f"Observed foliar symptoms are characteristic of {resolved_crop} {resolved_disease}.",
            "weather_interpretation": "Maintain regular monitoring under prevailing regional conditions.",
            "management_guidance": "Apply ICAR-recommended cultural practices and consult local agro-dealers for registered fungicides.",
            "prevention": "Prune affected leaves, avoid overhead irrigation, and sanitize tools.",
            "precautions": "Wear personal protective equipment (PPE) when handling chemical solutions.",
            "uncertainty_warning": None,
            "expert_advisory": "Contact your nearest Krishi Vigyan Kendra (KVK) or Block Agriculture Officer for on-field verification.",
            "language": language,
            "powered_by": "ICAR Grounded Advisory",
        }

    guidance_en = guidance.get("en", guidance)
    guidance_mr = guidance.get("mr", guidance)
    direct_answer_en = (
        guidance.get("direct_answer_en")
        or (guidance_en.get("direct_answer") if isinstance(guidance_en, dict) else "")
        or guidance.get("direct_answer", "")
    )
    direct_answer_mr = (
        guidance.get("direct_answer_mr")
        or (guidance_mr.get("direct_answer") if isinstance(guidance_mr, dict) else "")
        or guidance.get("direct_answer", "")
    )
    final_direct_answer = direct_answer_mr if is_mr and direct_answer_mr else direct_answer_en

    return {
        "gemini_guidance": guidance,
        "guidance_en": guidance_en,
        "guidance_mr": guidance_mr,
        "direct_answer": final_direct_answer,
        "direct_answer_en": direct_answer_en,
        "direct_answer_mr": direct_answer_mr,
        "crop": resolved_crop,
        "crop_en": crop_en,
        "crop_mr": crop_mr,
        "disease": resolved_disease,
        "disease_en": disease_en,
        "disease_mr": disease_mr,
        "execution_status": "success",
        "completed_stages": ["gemini_guidance"],
    }

