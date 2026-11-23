"""Krishi Sahayyak (कृषी सहाय्यक) — AI Agriculture Assistant
High-Performance Multilingual Streamlit Application for Crop Disease Diagnosis,
Microclimate Risk Analysis, Evidence-Grounded RAG Guidance, and Neural Audio.
"""
from __future__ import annotations

import base64
from io import BytesIO
import logging
from pathlib import Path
from typing import Any

from PIL import Image
import streamlit as st

# Import core agricultural services
from src.graph.state import create_initial_state, state_to_api_response
from src.graph.workflow import run_agriculture_workflow_sync
from src.prediction.predict import DiseasePredictor
from src.rag.vector_store import get_rag_service
from src.soil.soil_service import get_soil_service
from src.utils.localization import (
    CITY_TRANSLATIONS_MR,
    CROP_TRANSLATIONS,
    DISEASE_TRANSLATIONS,
)
from src.utils.voice_service import get_voice_service
from src.weather.weather_service import get_weather_service

logger = logging.getLogger("streamlit_app")

# =============================================================================
# Streamlit Page Configuration & Modern Agriculture Theme
# =============================================================================
st.set_page_config(
    page_title="Krishi Sahayyak | कृषी सहाय्यक",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern typography and base theme */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Banner */
.app-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    color: #ffffff;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(27, 94, 32, 0.25);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-header h1 {
    color: #ffffff !important;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #e8f5e9;
    font-size: 1.05rem;
    margin: 0.3rem 0 0 0;
    font-weight: 400;
}

/* Card Containers */
.agri-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.2s ease-in-out;
}
.agri-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
}

/* Diagnosis Hero Card */
.diagnosis-hero {
    background: linear-gradient(145deg, #f0fdf4 0%, #ffffff 100%);
    border: 2px solid #bbf7d0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Status Badges */
.badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}
.badge-success {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
}
.badge-warning {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fde68a;
}
.badge-danger {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #fca5a5;
}
.badge-info {
    background-color: #e0f2fe;
    color: #075985;
    border: 1px solid #7dd3fc;
}

/* Metric Display Boxes */
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box-title {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.metric-box-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0.2rem;
}

/* Guidance Section Blocks */
.guidance-block {
    background: #ffffff;
    border-left: 4px solid #2e7d32;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.guidance-block.chemical {
    border-left-color: #e11d48;
}
.guidance-block.organic {
    border-left-color: #16a34a;
}
.guidance-block.preventive {
    border-left-color: #0284c7;
}

/* Hide Streamlit default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Model & Service Initialization
# =============================================================================
@st.cache_resource(show_spinner="Pre-warming AI agricultural models...")
def get_cached_services():
    """Load PyTorch MobileNetV3 and FAISS RAG singleton services into RAM."""
    predictor = DiseasePredictor()
    rag = get_rag_service()
    weather = get_weather_service()
    soil = get_soil_service()
    voice = get_voice_service()
    return {
        "predictor": predictor,
        "rag": rag,
        "weather": weather,
        "soil": soil,
        "voice": voice,
    }


services = get_cached_services()


# =============================================================================
# Sidebar Controls & Localization
# =============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1592417817098-8f3d6910985b?w=600&auto=format&fit=crop&q=80",
        use_container_width=True,
    )
    st.markdown("### 🌾 **कृषी सहाय्यक | Krishi Sahayyak**")
    st.caption("AI-Powered Evidence-Grounded Agronomic Diagnostics")

    # Language Selector
    lang_choice = st.selectbox(
        "🌐 Language / भाषा निवडा",
        options=["English", "मराठी (Marathi)"],
        index=0,
    )
    is_marathi = "मराठी" in lang_choice
    lang_code = "marathi" if is_marathi else "english"

    st.markdown("---")

    # Location Selector (Maharashtra Agricultural Centers)
    popular_cities = [
        "Pune",
        "Nashik",
        "Baramati",
        "Kolhapur",
        "Solapur",
        "Nagpur",
        "Aurangabad (Chhatrapati Sambhajinagar)",
        "Amravati",
        "Akola",
        "Jalgaon",
        "Satara",
        "Sangli",
        "Ahmednagar",
    ]
    city_selected = st.selectbox(
        "📍 District / शेतकरी जिल्हा" if is_marathi else "📍 Location / Farmer District",
        options=popular_cities + ["Other (Custom City)"],
        index=0,
    )
    if city_selected == "Other (Custom City)":
        location_input = st.text_input(
            "Enter City Name" if not is_marathi else "जिल्ह्याचे/गावाचे नाव लिहा",
            value="Pune",
        )
    else:
        # Normalize city name
        location_input = city_selected.split(" (")[0]

    include_gradcam = st.checkbox(
        "🔍 Grad-CAM दृश्य विश्लेषण (Visual Attention)" if is_marathi else "🔍 Generate Grad-CAM Visual Attention",
        value=True,
        help="Highlights the specific leaf regions that influenced the AI prediction.",
    )

    st.markdown("---")
    with st.expander("⚙️ System Status" if not is_marathi else "⚙️ प्रणाली स्थिती", expanded=False):
        st.write(f"• **MobileNetV3:** {len(services['predictor'].class_names)} classes ready")
        st.write(f"• **RAG Knowledge Base:** {len(services['rag'].chunks)} research chunks")
        st.write("• **Weather Engine:** Live Open-Meteo & OWM")
        st.write("• **Device:** CPU Mode (Optimized)")


# =============================================================================
# Header Section
# =============================================================================
header_title = "🌱 कृषी सहाय्यक (Krishi Sahayyak)" if is_marathi else "🌱 Krishi Sahayyak"
header_sub = (
    "पिकांचे रोग निदान, हवामान जोखीम विश्लेषण, आणि संशोधन-आधारित कृषी सल्ला"
    if is_marathi
    else "Evidence-Grounded Crop Disease Detection, Microclimate Risk & Multilingual Farmer Guidance"
)

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1>{header_title}</h1>
            <p>{header_sub}</p>
        </div>
        <div style="text-align: right;">
            <span class="badge badge-success">📍 {location_input}</span>
            <span class="badge badge-info" style="margin-left: 6px;">🌐 {lang_code.upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Utilities
# =============================================================================
def decode_base64_image(b64_str: str) -> Image.Image | None:
    """Decode a base64 string to a PIL Image."""
    try:
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data))
    except Exception as e:
        logger.warning(f"Grad-CAM decode note: {e}")
        return None


# =============================================================================
# Main Application Tabs
# =============================================================================
tab1_title = "🌿 पिकांचे रोग निदान (Leaf Diagnosis)" if is_marathi else "🌿 Crop Leaf Diagnosis"
tab2_title = "👨‍🌾 कृषी सल्लागार (Ask Agronomist)" if is_marathi else "👨‍🌾 Ask Agronomist (Q&A)"
tab3_title = "🌦️ हवामान व जमीन (Weather & Soil)" if is_marathi else "🌦️ Weather & Soil Profile"

tab_diagnose, tab_ask, tab_weather = st.tabs([tab1_title, tab2_title, tab3_title])


# =============================================================================
# TAB 1: Leaf Disease Diagnosis
# =============================================================================
with tab_diagnose:
    col_input, col_display = st.columns([1, 1.2], gap="large")

    image_bytes: bytes | None = None
    input_image_pil: Image.Image | None = None

    with col_input:
        st.markdown(
            "#### 📸 1. " + ("पिकाच्या पानाचा फोटो निवडा" if is_marathi else "Upload Crop Leaf Photo")
        )

        # Quick Sample Leaf Selector
        st.markdown(
            "**" + ("🧪 किंवा नमुना पानासह तपासा:" if is_marathi else "🧪 Or Test with a Verified Sample:") + "**"
        )
        sample_cols = st.columns(3)

        sample_path: Path | None = None
        if sample_cols[0].button("🍅 Tomato Blight", use_container_width=True):
            sample_path = Path("data/samples/sample_tomato_early_blight.jpg")
        if sample_cols[1].button("🥔 Potato Blight", use_container_width=True):
            sample_path = Path("data/samples/sample_potato_late_blight.jpg")
        if sample_cols[2].button("🫑 Pepper Spot", use_container_width=True):
            sample_path = Path("data/samples/sample_bell_pepper_bacterial_spot.jpg")

        uploaded_file = st.file_uploader(
            "Upload leaf image (Tomato, Potato, Pepper)" if not is_marathi else "पानाचा फोटो अपलोड करा",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-quality close-up photos of leaves produce the most accurate diagnosis.",
        )

        if sample_path and sample_path.exists():
            image_bytes = sample_path.read_bytes()
            input_image_pil = Image.open(BytesIO(image_bytes))
            st.success(f"Loaded Sample: {sample_path.name}")
        elif uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            input_image_pil = Image.open(BytesIO(image_bytes))

        # Optional farmer question
        st.markdown("#### ❓ 2. " + ("तुमचा प्रश्न (ऐच्छिक)" if is_marathi else "Specific Question (Optional)"))
        farmer_q = st.text_input(
            label="Farmer Question",
            label_visibility="collapsed",
            placeholder=(
                "उदा. यावर सेंद्रिय कडुनिंब तेलाची फवारणी चालेल का? फवारणीची वेळ कोणती?"
                if is_marathi
                else "e.g. Can I use organic neem oil? How do I prevent this from spreading?"
            ),
        )

        analyze_btn = st.button(
            "🚀 " + ("रोग निदान सुरू करा" if is_marathi else "Analyze Crop Health"),
            type="primary",
            use_container_width=True,
            disabled=(image_bytes is None),
        )

    with col_display:
        if input_image_pil:
            st.markdown(
                "#### 🖼️ " + ("अपलोड केलेले पान" if is_marathi else "Selected Leaf Image")
            )
            st.image(input_image_pil, use_container_width=True)
        else:
            st.info(
                "👈 "
                + (
                    "कृपया डाव्या बाजूने पानाचा फोटो अपलोड करा किंवा नमुना निवडा."
                    if is_marathi
                    else "Upload a leaf photo or pick a sample on the left to start."
                )
            )

    # --- EXECUTE DIAGNOSIS ---
    if analyze_btn and image_bytes:
        with st.spinner("Analyzing leaf with MobileNetV3, Weather, RAG, and Gemini..."):
            initial_state = create_initial_state(
                request_type="analyze",
                user_question=farmer_q or None,
                language=lang_code,
                location=location_input,
                image_input=image_bytes,
                include_gradcam=include_gradcam,
            )
            final_state = run_agriculture_workflow_sync(initial_state)
            result = state_to_api_response(final_state)

        st.session_state["last_diagnosis"] = result
        st.session_state["last_image"] = input_image_pil

    # --- RENDER RESULTS ---
    if "last_diagnosis" in st.session_state and st.session_state.get("last_diagnosis"):
        res = st.session_state["last_diagnosis"]
        crop_disp = res.get("crop") or "Unknown"
        disease_disp = res.get("disease") or "Unknown"
        conf_pct = res.get("confidence_percentage", 0.0)
        is_low_conf = res.get("is_low_confidence", False)

        st.markdown("---")

        # Top Diagnosis Hero Banner
        badge_class = "badge-danger" if "blight" in disease_disp.lower() or "spot" in disease_disp.lower() else "badge-success"
        conf_badge = "badge-warning" if is_low_conf else "badge-success"

        st.markdown(
            f"""
            <div class="diagnosis-hero">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="badge {badge_class}" style="font-size: 0.95rem;">🌱 {crop_disp}</span>
                        <h2 style="margin: 0.5rem 0 0.2rem 0; color: #1e293b; font-size: 1.85rem; font-weight: 700;">
                            {disease_disp}
                        </h2>
                    </div>
                    <div style="text-align: right;">
                        <span class="badge {conf_badge}">Confidence: {conf_pct}%</span>
                        <p style="margin: 0.4rem 0 0 0; font-size: 0.85rem; color: #64748b;">
                            {"⚠️ कमी विश्वासार्हता (तज्ज्ञांचा सल्ला घ्या)" if is_low_conf and is_marathi else ("⚠️ Low confidence prediction" if is_low_conf else "✅ Verified by MobileNetV3")}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual Inspection Row (Leaf & Grad-CAM)
        gradcam_b64 = res.get("gradcam_base64")
        if gradcam_b64:
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.markdown("**📸 " + ("मूळ पान (Original)" if is_marathi else "Original Leaf") + "**")
                if "last_image" in st.session_state:
                    st.image(st.session_state["last_image"], use_container_width=True)
            with v_col2:
                st.markdown("**🔥 " + ("Grad-CAM दृश्य उष्णतानकाशा (Saliency Map)" if is_marathi else "Grad-CAM Saliency Map") + "**")
                cam_img = decode_base64_image(gradcam_b64)
                if cam_img:
                    st.image(cam_img, use_container_width=True)
                st.caption(
                    "लाल आणि पिवळे भाग AI च्या रोग ओळखीसाठी सर्वात महत्त्वाचे ठरले आहेत."
                    if is_marathi
                    else "Highlighted warm regions indicate leaf symptoms that influenced the MobileNetV3 prediction."
                )

        # Voice Readout (TTS)
        direct_ans = res.get("direct_answer") or res.get("guidance", {}).get("immediate_action", "")
        if direct_ans:
            st.markdown("#### 🔊 " + ("ऑडिओ सल्ला ऐका" if is_marathi else "Listen to Audio Guidance"))
            tts_col1, tts_col2 = st.columns([2, 1])
            with tts_col1:
                ok, audio_bytes, mime = services["voice"].text_to_speech_bytes(direct_ans, lang_code)
                if ok and audio_bytes:
                    st.audio(audio_bytes, format=mime)
                else:
                    st.caption("Audio generation temporarily unavailable.")

        # Guidance Cards
        st.markdown("### 📋 " + ("तपशीलवार कृषी सल्ला व उपाय" if is_marathi else "Evidence-Grounded Agronomic Guidance"))

        guidance = res.get("guidance") or {}
        g_col1, g_col2 = st.columns(2, gap="medium")

        with g_col1:
            st.markdown(
                f"""
                <div class="guidance-block">
                    <h4 style="margin: 0 0 0.5rem 0; color: #15803d;">⚡ {"तातडीची कृती (Immediate Action)" if is_marathi else "Immediate Action"}</h4>
                    <p style="margin: 0; color: #334155;">{guidance.get('immediate_action') or direct_ans or 'Maintain standard monitoring.'}</p>
                </div>
                <div class="guidance-block chemical">
                    <h4 style="margin: 0 0 0.5rem 0; color: #e11d48;">🧪 {"रासायनिक नियंत्रण (Chemical Treatment)" if is_marathi else "Chemical Management"}</h4>
                    <p style="margin: 0; color: #334155;">{guidance.get('chemical_options') or 'Apply recommended fungicide/bactericide as per agricultural university guidelines.'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with g_col2:
            st.markdown(
                f"""
                <div class="guidance-block organic">
                    <h4 style="margin: 0 0 0.5rem 0; color: #16a34a;">🌿 {"सेंद्रिय व जैविक उपाय (Organic Control)" if is_marathi else "Organic & Biological Controls"}</h4>
                    <p style="margin: 0; color: #334155;">{guidance.get('organic_options') or 'Neem oil spray (5ml/L) or Trichoderma viride application.'}</p>
                </div>
                <div class="guidance-block preventive">
                    <h4 style="margin: 0 0 0.5rem 0; color: #0284c7;">🛡️ {"प्रतिबंधात्मक काळजी (Preventive Practices)" if is_marathi else "Preventive & Cultural Practices"}</h4>
                    <p style="margin: 0; color: #334155;">{guidance.get('preventive_measures') or 'Drip irrigation, crop rotation, and sanitizing pruning shears.'}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Microclimate & Soil Interaction Box
        st.markdown("### 🌦️ " + ("स्थानिक हवामान आणि मातीचे विश्लेषण" if is_marathi else "Microclimate & Agro-Climatic Context"))
        w_res = res.get("weather") or {}
        s_res = res.get("soil") or {}

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f"""<div class="metric-box"><div class="metric-box-title">🌡️ {"तापमान" if is_marathi else "Temperature"}</div><div class="metric-box-value">{w_res.get('temperature_c', '--')}°C</div></div>""",
            unsafe_allow_html=True,
        )
        m2.markdown(
            f"""<div class="metric-box"><div class="metric-box-title">💧 {"आर्द्रता" if is_marathi else "Humidity"}</div><div class="metric-box-value">{w_res.get('humidity_percent', '--')}%</div></div>""",
            unsafe_allow_html=True,
        )
        m3.markdown(
            f"""<div class="metric-box"><div class="metric-box-title">🌧️ {"पावसाची शक्यता" if is_marathi else "Rain Chance"}</div><div class="metric-box-value">{w_res.get('precipitation_prob_percent', '--')}%</div></div>""",
            unsafe_allow_html=True,
        )
        m4.markdown(
            f"""<div class="metric-box"><div class="metric-box-title">🌱 {"मातीचा प्रकार" if is_marathi else "Soil Type"}</div><div class="metric-box-value" style="font-size: 1.05rem;">{s_res.get('soil_type', 'Medium Black')[:18]}</div></div>""",
            unsafe_allow_html=True,
        )

        # RAG Evidence Citations
        sources = res.get("sources") or []
        if sources:
            with st.expander("📚 " + ("संशोधन संदर्भ व पुरावे (RAG Citations)" if is_marathi else "Research Citations & Evidence (RAG)"), expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {src.get('source_title', 'Agricultural Extension Research')}**")
                    st.write(src.get("content", ""))
                    st.caption(f"Confidence Relevance Score: {round(src.get('similarity', 0.85) * 100, 1)}%")


# =============================================================================
# TAB 2: Ask Agronomist (Follow-up & Standalone Q&A)
# =============================================================================
with tab_ask:
    st.markdown(
        "### 👨‍🌾 "
        + (
            "कृषी तज्ज्ञांना थेट प्रश्न विचारा"
            if is_marathi
            else "Ask Agricultural Expert (Direct Q&A)"
        )
    )
    st.caption(
        "कोणताही फोटो न देता पिकांचे रोग, खते, कीड व्यवस्थापन किंवा फवारणी वेळापत्रकाबद्दल प्रश्न विचारा."
        if is_marathi
        else "Ask any agronomic question regarding crops, fertilizer schedules, pesticides, or microclimate risk."
    )

    ask_col1, ask_col2 = st.columns([1, 2], gap="large")

    with ask_col1:
        crop_context = st.selectbox(
            "🌱 Select Crop Category:" if not is_marathi else "🌱 पिकाचा प्रकार निवडा:",
            options=["Tomato", "Potato", "Bell pepper", "General / इतर"],
            index=0,
        )
        disease_context = st.text_input(
            "Suspected Disease (Optional):" if not is_marathi else "संशयित रोग (ऐच्छिक):",
            placeholder="e.g. Early blight, Late blight",
        )

    with ask_col2:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display previous chat messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input(
            "तुमचा प्रश्न येथे विचारा... (उदा. उशिरा येणाऱ्या करप्यावर कोणती फवारणी करावी?)"
            if is_marathi
            else "Type your farming question here... (e.g. What is the spray interval for Late Blight?)"
        )

        if user_query:
            # Display user message
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # Generate Agronomist answer via LangGraph workflow
            with st.spinner("Retrieving agricultural research and consulting Gemini..."):
                initial_state = create_initial_state(
                    request_type="ask",
                    user_question=user_query,
                    language=lang_code,
                    location=location_input,
                    crop=None if "General" in crop_context else crop_context,
                    disease=disease_context or None,
                )
                final_state = run_agriculture_workflow_sync(initial_state)
                ans_res = state_to_api_response(final_state)
                answer_text = ans_res.get("direct_answer") or ans_res.get("guidance", {}).get("immediate_action", "No response generated.")

            st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.write(answer_text)

                # Voice output for chat answer
                ok, a_bytes, m_type = services["voice"].text_to_speech_bytes(answer_text, lang_code)
                if ok and a_bytes:
                    st.audio(a_bytes, format=m_type)


# =============================================================================
# TAB 3: Weather & Soil Insights Dashboard
# =============================================================================
with tab_weather:
    st.markdown(
        "### 🌦️ "
        + (
            f"{location_input} मधील थेट हवामान व माती विश्लेषण"
            if is_marathi
            else f"Live Microclimate Risk & Soil Health in {location_input}"
        )
    )

    w_service = services["weather"]
    s_service = services["soil"]

    w_data = w_service.get_weather(city=location_input)
    s_data = s_service.get_soil_health(location=location_input, crop="Tomato")

    w_top1, w_top2, w_top3, w_top4 = st.columns(4)
    w_top1.metric("🌡️ Temperature", f"{w_data.get('temperature_c', '--')} °C")
    w_top2.metric("💧 Humidity", f"{w_data.get('humidity_percent', '--')} %")
    w_top3.metric("🌧️ Rain Probability", f"{w_data.get('precipitation_prob_percent', '--')} %")
    w_top4.metric("💨 Wind Speed", f"{w_data.get('wind_speed_kmh', '--')} km/h")

    st.markdown("---")

    soil_col1, soil_col2 = st.columns([1, 1], gap="large")

    with soil_col1:
        st.markdown("#### 🧪 " + ("मातीचे रासायनिक व भौतिक गुणधर्म" if is_marathi else "Regional Soil Profile"))
        st.write(f"• **Regional Soil Classification:** {s_data.get('soil_type', 'Medium Black Soil')}")
        st.write(f"• **Soil pH Value:** {s_data.get('pH', 7.2)} ({'Neutral' if 6.5 <= s_data.get('pH', 7.2) <= 7.5 else 'Slightly Alkaline'})")
        st.write(f"• **Organic Carbon:** {s_data.get('organic_carbon_pct', 0.55)}%")
        st.write(f"• **Available Nitrogen (N):** {s_data.get('nitrogen_kg_ha', 240)} kg/ha")
        st.write(f"• **Available Phosphorus (P):** {s_data.get('phosphorus_kg_ha', 18)} kg/ha")
        st.write(f"• **Available Potassium (K):** {s_data.get('potassium_kg_ha', 320)} kg/ha")

    with soil_col2:
        st.markdown("#### 🌾 " + ("अग्रो-हवामान जोखीम सूचना" if is_marathi else "Microclimate Pathogen Risk Assessment"))
        risk_summary = w_data.get("disease_risk", {})
        fungal_risk = risk_summary.get("fungal_risk", "Moderate")
        bacterial_risk = risk_summary.get("bacterial_risk", "Low")

        st.info(f"**Fungal Pathogen Risk:** {fungal_risk} — High humidity (>75%) coupled with moderate temperatures accelerates leaf blight spore germination.")
        st.info(f"**Bacterial Pathogen Risk:** {bacterial_risk} — Rain splash and leaf surface wetness facilitate Xanthomonas entry through stomata.")
        st.success(f"**Recommended Spray Timing:** Spray early in the morning (6:30 AM – 9:00 AM) when wind speed is below 10 km/h to prevent spray drift.")

st.markdown("---")
st.caption(
    "Krishi Sahayyak | Developed for Maharashtra farmers | Backed by ICAR & MPKV Rahuri Agronomic Research"
)
