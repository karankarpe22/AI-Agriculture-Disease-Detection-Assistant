"""LangGraph workflow assembly and routing for the AI Agriculture Assistant.

Assembles and compiles the StateGraph workflow orchestrating:
1. End-to-end leaf image disease diagnosis (/analyze)
2. Follow-up and standalone agronomist question answering (/ask)

Supports parallel execution of context retrieval services (weather, soil, RAG),
conditional Grad-CAM execution without race conditions, and early termination on quality failure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    context_aggregation_node,
    gemini_guidance_node,
    gradcam_node,
    prediction_node,
    quality_node,
    rag_node,
    soil_node,
    weather_node,
)
from src.graph.state import AgricultureAssistantState
from src.utils.localization import CROP_TRANSLATIONS, DISEASE_TRANSLATIONS

logger = logging.getLogger("src.graph.workflow")


# =============================================================================
# Standalone Ask Context Setup Node
# =============================================================================
async def setup_ask_context(state: AgricultureAssistantState) -> dict[str, Any]:
    """Resolve crop and disease context for standalone or follow-up question answering."""
    crop = state.get("crop")
    disease = state.get("disease")
    question = state.get("user_question") or ""

    # Normalize user-provided crop across casing and Marathi script
    if crop:
        c_clean = str(crop).strip().lower()
        if c_clean in ["tomato", "टोमॅटो"]:
            crop = "Tomato"
        elif c_clean in ["potato", "बटाटा"]:
            crop = "Potato"
        elif any(k in c_clean for k in ["pepper", "bell pepper", "capsicum", "शिमला", "मिरची", "ढोबळी मिरची"]):
            crop = "Bell pepper"

    # Infer crop context from natural language question if not explicitly provided
    if not crop and question:
        q_lower = question.lower()
        if "tomato" in q_lower or "टोमॅटो" in q_lower:
            crop = "Tomato"
        elif "potato" in q_lower or "बटाटा" in q_lower:
            crop = "Potato"
        elif any(k in q_lower for k in ["pepper", "capsicum", "शिमला", "मिरची"]):
            crop = "Bell pepper"

    # Normalize user-provided disease across casing and Marathi script
    if disease:
        d_clean = str(disease).strip().lower()
        if "early blight" in d_clean or "लवकर येणारा करपा" in d_clean:
            disease = "Early blight"
        elif "late blight" in d_clean or "उशिरा येणारा करपा" in d_clean:
            disease = "Late blight"
        elif "bacterial spot" in d_clean or "जिवाणूजन्य" in d_clean:
            disease = "Bacterial spot"

    # Infer disease context from natural language question if not explicitly provided
    if not disease and question:
        q_lower = question.lower()
        if "early blight" in q_lower or "लवकर येणारा करपा" in q_lower:
            disease = "Early blight"
        elif "late blight" in q_lower or "उशिरा येणारा करपा" in q_lower:
            disease = "Late blight"
        elif "bacterial spot" in q_lower or "जिवाणूजन्य" in q_lower:
            disease = "Bacterial spot"

    resolved_crop = crop or "General / Solanaceae"
    resolved_disease = disease or "General Agronomic Inquiry"

    crop_en = CROP_TRANSLATIONS.get(resolved_crop, {}).get("en", resolved_crop)
    crop_mr = CROP_TRANSLATIONS.get(resolved_crop, {}).get("mr", resolved_crop)
    disease_en = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("en", resolved_disease)
    disease_mr = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("mr", resolved_disease)

    return {
        "crop": resolved_crop,
        "crop_en": crop_en,
        "crop_mr": crop_mr,
        "disease": resolved_disease,
        "disease_en": disease_en,
        "disease_mr": disease_mr,
        "completed_stages": ["ask_context_setup"],
    }


# =============================================================================
# Routing Functions
# =============================================================================
def route_start(state: AgricultureAssistantState) -> str:
    """Route incoming request by type: 'analyze' goes to quality check, 'ask' goes to context setup."""
    req_type = state.get("request_type", "analyze")
    if req_type == "analyze":
        return "quality_node"
    return "setup_ask_context"


def route_quality(state: AgricultureAssistantState) -> str:
    """Route after quality check: abort directly to END on failure; proceed to prediction on success."""
    if state.get("execution_status") == "quality_failed":
        return END
    return "prediction_node"


def route_prediction(state: AgricultureAssistantState) -> list[str] | str:
    """Fan out after prediction to parallel context services; include gradcam conditionally."""
    if state.get("execution_status") == "error":
        return END

    branches = ["weather_node", "soil_node", "rag_node"]
    if state.get("include_gradcam"):
        branches.append("gradcam_node")
    return branches


# =============================================================================
# Graph Builder
# =============================================================================
def build_agriculture_graph() -> Any:
    """Construct and compile the LangGraph StateGraph for the AI Agriculture Assistant."""
    builder = StateGraph(AgricultureAssistantState)

    # 1. Register all nodes
    builder.add_node("quality_node", quality_node)
    builder.add_node("prediction_node", prediction_node)
    builder.add_node("setup_ask_context", setup_ask_context)
    builder.add_node("weather_node", weather_node)
    builder.add_node("soil_node", soil_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("gradcam_node", gradcam_node)
    builder.add_node("context_aggregation_node", context_aggregation_node)
    builder.add_node("gemini_guidance_node", gemini_guidance_node)

    # 2. Conditional entry from START
    builder.add_conditional_edges(
        START,
        route_start,
        {
            "quality_node": "quality_node",
            "setup_ask_context": "setup_ask_context",
        },
    )

    # 3. Conditional exit from quality_node on quality check failure
    builder.add_conditional_edges(
        "quality_node",
        route_quality,
        {
            "prediction_node": "prediction_node",
            END: END,
        },
    )

    # 4. Fan-out from prediction_node to parallel context services + optional gradcam
    builder.add_conditional_edges(
        "prediction_node",
        route_prediction,
        ["weather_node", "soil_node", "rag_node", "gradcam_node", END],
    )

    # 5. Fan-out from setup_ask_context to parallel context services (no CV or gradcam)
    builder.add_edge("setup_ask_context", "weather_node")
    builder.add_edge("setup_ask_context", "soil_node")
    builder.add_edge("setup_ask_context", "rag_node")

    # 6. Join parallel context branches into context_aggregation_node
    builder.add_edge("weather_node", "context_aggregation_node")
    builder.add_edge("soil_node", "context_aggregation_node")
    builder.add_edge("rag_node", "context_aggregation_node")
    builder.add_edge("gradcam_node", "context_aggregation_node")

    # 7. Aggregation to Gemini advisory generation to END
    builder.add_edge("context_aggregation_node", "gemini_guidance_node")
    builder.add_edge("gemini_guidance_node", END)

    return builder.compile()


# Module-level compiled instance for application reuse
agriculture_graph = build_agriculture_graph()


# =============================================================================
# Execution Helpers
# =============================================================================
async def run_agriculture_workflow(initial_state: AgricultureAssistantState) -> AgricultureAssistantState:
    """Execute the compiled LangGraph workflow asynchronously."""
    final_state = await agriculture_graph.ainvoke(initial_state)
    return final_state


def run_agriculture_workflow_sync(initial_state: AgricultureAssistantState) -> AgricultureAssistantState:
    """Synchronous execution helper for testing or CLI invocation."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, agriculture_graph.ainvoke(initial_state)).result()

    return asyncio.run(agriculture_graph.ainvoke(initial_state))

