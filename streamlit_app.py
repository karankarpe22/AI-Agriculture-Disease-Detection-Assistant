"""Krishi Sahayyak (कृषी सहाय्यक) — Premium Multilingual AI Agriculture Assistant.
Evidence-Grounded Crop Disease Detection, Microclimate Risk & Dual-Language Agronomic Advisory.
"""
from __future__ import annotations

import base64
from io import BytesIO
import logging
from pathlib import Path
from typing import Any

from PIL import Image
import streamlit as st

# Core agricultural services
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
# Streamlit Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Krishi Sahayyak | कृषी सहाय्यक",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# High-End Modern Agritech CSS Design System
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

/* Global Font & Reset */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', -apple-system, sans-serif;
}

/* Modern Header Hero Banner */
.app-hero-banner {
    background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #10b981 100%);
    border-radius: 20px;
    padding: 2rem 2.4rem;
    color: #ffffff;
    margin-bottom: 1.8rem;
    box-shadow: 0 15px 35px -5px rgba(6, 78, 59, 0.35);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.03em;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #d1fae5;
    margin: 0.4rem 0 0 0;
    font-weight: 400;
    line-height: 1.4;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 0.4rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #ffffff;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Universal Section Card */
.ks-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.ks-card:hover {
    box-shadow: 0 8px 25px -4px rgba(15, 23, 42, 0.08);
}

/* Section Header Badges */
.sec-badge-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #f1f5f9;
}
.sec-icon-circle {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    background: #ecfdf5;
    color: #047857;
}
.sec-title-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #0f172a;
    margin: 0;
}

/* Diagnosis Hero Banner */
.diag-hero-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f0fdf4 100%);
    border: 2px solid #86efac;
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 20px -3px rgba(16, 185, 129, 0.12);
}
.diag-crop-tag {
    background: #047857;
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
}
.diag-disease-name {
    font-size: 2rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.6rem 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.conf-gauge-container {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 10px;
    width: 100%;
    margin-top: 0.6rem;
    overflow: hidden;
}
.conf-gauge-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #10b981, #047857);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Structured Gemini Guidance Blocks */
.guidance-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.guidance-box.summary {
    border-left-color: #6366f1;
    background: linear-gradient(145deg, #fdf4ff 0%, #ffffff 100%);
    border-color: #e0e7ff;
}
.guidance-box.pathology {
    border-left-color: #0284c7;
    background: #f8fafc;
}
.guidance-box.treatment {
    border-left-color: #059669;
    background: #f0fdf4;
}
.guidance-box.prevention {
    border-left-color: #0d9488;
    background: #f0fdfa;
}
.guidance-box.precautions {
    border-left-color: #f59e0b;
    background: #fffbeb;
}
.guidance-box.weather-impact {
    border-left-color: #3b82f6;
    background: #eff6ff;
}
.guidance-box.soil-impact {
    border-left-color: #d97706;
    background: #fffbeb;
}
.guidance-box.advisory {
    border-left-color: #8b5cf6;
    background: #faf5ff;
}

.guidance-box-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    color: #1e293b;
}

/* Metric Display Tiles */
.ks-metric-tile {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.ks-metric-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}
.ks-metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #0f172a;
}
.ks-metric-sub {
    font-size: 0.8rem;
    font-weight: 500;
    color: #10b981;
    margin-top: 0.2rem;
}

/* Image Frame */
.image-card-frame {
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    overflow: hidden;
    background: #0f172a;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

/* Quick Chips */
.chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 0.6rem 0 1rem 0;
}

/* Clean Streamlit Components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stSidebarNav"] {display: none;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Model & Service Initialization
# =============================================================================
@st.cache_resource(show_spinner="Pre-warming AI agricultural models into RAM...")
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
# Sidebar Controls & Language Toggle
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.8rem 0 1.2rem 0;">
            <div style="font-size: 3rem; line-height: 1;">🌱</div>
            <h2 style="margin: 0.4rem 0 0.1rem 0; font-weight: 800; color: #064e3b; font-size: 1.6rem;">कृषी सहाय्यक</h2>
            <div style="color: #64748b; font-size: 0.88rem; font-weight: 500;">AI Agriculture Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Language Switcher
    lang_choice = st.selectbox(
        "🌐 Language / भाषा निवडा",
        options=["English", "मराठी (Marathi)"],
        index=0,
    )
    is_marathi = "मराठी" in lang_choice
    lang_code = "marathi" if is_marathi else "english"

    st.markdown("---")

    # District Selector
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
        location_input = city_selected.split(" (")[0]

    # Grad-CAM Heatmap Toggle
    include_gradcam = st.checkbox(
        "🔍 Grad-CAM दृश्य उष्णतानकाशा (Heatmap)" if is_marathi else "🔍 Generate Grad-CAM Heatmap",
        value=True,
        help="Highlights the exact leaf disease symptoms that influenced the MobileNetV3 AI decision.",
    )

    st.markdown("---")
    with st.expander("⚙️ System Status" if not is_marathi else "⚙️ प्रणाली स्थिती", expanded=False):
        st.write(f"• **MobileNetV3:** {len(services['predictor'].class_names)} Classes Ready")
        st.write(f"• **RAG Knowledge Base:** {len(services['rag'].chunks)} Research Chunks")
        st.write("• **Weather Microclimate:** Live Open-Meteo & OWM")
        st.write("• **Soil Health Profile:** Synthetic Regional Records")
        st.write("• **Device:** CPU Mode (Optimized 512MB+)")


# =============================================================================
# Header Hero Banner
# =============================================================================
header_title = "कृषी सहाय्यक (Krishi Sahayyak)" if is_marathi else "Krishi Sahayyak"
header_sub = (
    "पिकांचे रोग निदान, स्थानिक हवामान व माती विश्लेषण, आणि संशोधन-आधारित कृषी सल्लागार"
    if is_marathi
    else "Evidence-Grounded Crop Disease Detection, Microclimate Risk & Multilingual Farmer Guidance"
)

st.markdown(
    f"""
    <div class="app-hero-banner">
        <div>
            <div class="hero-title">🌱 {header_title}</div>
            <div class="hero-subtitle">{header_sub}</div>
        </div>
        <div class="hero-badges">
            <span class="hero-pill">📍 {location_input}</span>
            <span class="hero-pill">🌐 {lang_code.upper()}</span>
            <span class="hero-pill">⚡ MobileNetV3 + RAG</span>
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
# Navigation Tabs
# =============================================================================
tab1_title = "🌿 पिकांचे रोग निदान (Leaf Diagnosis)" if is_marathi else "🌿 Crop Leaf Diagnosis"
tab2_title = "👨‍🌾 कृषी सल्लागार (Ask Agronomist)" if is_marathi else "👨‍🌾 Ask Agronomist (Q&A)"
tab3_title = "🌦️ हवामान व जमीन (Weather & Soil)" if is_marathi else "🌦️ Weather & Soil Profile"

tab_diagnose, tab_ask, tab_weather = st.tabs([tab1_title, tab2_title, tab3_title])


# =============================================================================
# TAB 1: Leaf Disease Diagnosis
# =============================================================================
with tab_diagnose:
    col_input, col_preview = st.columns([1, 1.1], gap="large")

    image_bytes: bytes | None = None
    input_image_pil: Image.Image | None = None

    with col_input:
        st.markdown(
            "### 📸 1. " + ("पिकाच्या पानाचा फोटो अपलोड करा" if is_marathi else "Upload Crop Leaf Photo")
        )
        st.caption(
            "टोमॅटो, बटाटा किंवा ढोबळी मिरचीच्या पानाचा स्पष्ट फोटो अपलोड करा किंवा खालील नमुना निवडा."
            if is_marathi
            else "Upload a clear leaf photo (Tomato, Potato, Bell Pepper) or select a sample leaf below."
        )

        # Quick Sample Leaf Chips
        st.markdown(
            "**" + ("🧪 किंवा नमुना पानासह झटपट तपासा:" if is_marathi else "🧪 Or Test with a Verified Sample:") + "**"
        )
        s_col1, s_col2, s_col3 = st.columns(3)
        sample_path: Path | None = None

        if s_col1.button("🍅 Tomato Blight", use_container_width=True):
            sample_path = Path("data/samples/sample_tomato_early_blight.jpg")
        if s_col2.button("🥔 Potato Blight", use_container_width=True):
            sample_path = Path("data/samples/sample_potato_late_blight.jpg")
        if s_col3.button("🫑 Pepper Spot", use_container_width=True):
            sample_path = Path("data/samples/sample_bell_pepper_bacterial_spot.jpg")

        uploaded_file = st.file_uploader(
            "Select photo" if not is_marathi else "फोटो निवडा",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )

        if sample_path and sample_path.exists():
            image_bytes = sample_path.read_bytes()
            input_image_pil = Image.open(BytesIO(image_bytes))
            st.success(f"✅ Loaded sample: {sample_path.name}")
        elif uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            input_image_pil = Image.open(BytesIO(image_bytes))

        # Optional farmer question
        st.markdown("### ❓ 2. " + ("तुमचा प्रश्न (ऐच्छिक)" if is_marathi else "Specific Question (Optional)"))
        farmer_q = st.text_input(
            label="Farmer Question",
            label_visibility="collapsed",
            placeholder=(
                "उदा. यावर सेंद्रिय कडुनिंब तेलाची फवारणी चालेल का? फवारणीची वेळ कोणती?"
                if is_marathi
                else "e.g. Can I spray neem oil? What is the recommended fungicide dosage?"
            ),
        )

        analyze_btn = st.button(
            "🚀 " + ("रोग निदान सुरू करा (Analyze Health)" if is_marathi else "Analyze Crop Health"),
            type="primary",
            use_container_width=True,
            disabled=(image_bytes is None),
        )

    with col_preview:
        st.markdown("### 🖼️ " + ("पानाची पूर्वावलोकन प्रतिमा" if is_marathi else "Leaf Image Preview"))
        if input_image_pil:
            st.image(input_image_pil, use_container_width=True)
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #cbd5e1; border-radius: 16px; padding: 3rem 1.5rem; text-align: center; background: #f8fafc; color: #64748b;">
                    <div style="font-size: 2.8rem; margin-bottom: 0.5rem;">🌿</div>
                    <div style="font-weight: 600; font-size: 1.1rem; color: #334155;">No leaf image selected</div>
                    <div style="font-size: 0.9rem; margin-top: 0.3rem;">Upload a leaf photo or pick a sample on the left to begin diagnosis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- EXECUTE DIAGNOSIS WORKFLOW ---
    if analyze_btn and image_bytes:
        with st.spinner("🔬 Running MobileNetV3, Local Weather, RAG Retrieval, and Gemini Synthesis..."):
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

    # --- RENDER FULL ADVISORY RESULTS ---
    if "last_diagnosis" in st.session_state and st.session_state.get("last_diagnosis"):
        res = st.session_state["last_diagnosis"]
        crop_disp = res.get("crop") or "Crop"
        disease_disp = res.get("disease") or "Disease"
        conf_pct = res.get("confidence_percentage", 0.0)
        is_low_conf = res.get("is_low_confidence", False)

        # Extract structured guidance block based on active language
        guidance = res.get("guidance") or {}
        if is_marathi:
            g_lang = res.get("guidance_mr") or guidance.get("mr") or guidance
        else:
            g_lang = res.get("guidance_en") or guidance.get("en") or guidance

        direct_ans = g_lang.get("direct_answer") or res.get("direct_answer") or ""
        explanation_text = g_lang.get("explanation") or ""
        mgmt_text = g_lang.get("management_guidance") or ""
        prev_text = g_lang.get("prevention") or ""
        prec_text = g_lang.get("precautions") or ""
        weather_interp = g_lang.get("weather_interpretation") or ""
        soil_interp = g_lang.get("soil_interpretation") or ""
        expert_adv = g_lang.get("expert_advisory") or ""

        st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # 1. Top Diagnosis Banner
        st.markdown(
            f"""
            <div class="diag-hero-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="diag-crop-tag">🌱 {crop_disp}</span>
                        <div class="diag-disease-name">{disease_disp}</div>
                    </div>
                    <div style="text-align: right; min-width: 140px;">
                        <div style="font-size: 1.8rem; font-weight: 800; color: {'#d97706' if is_low_conf else '#059669'};">
                            {conf_pct}%
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #64748b; text-transform: uppercase;">
                            {"⚠️ Low Confidence" if is_low_conf else "✅ High Confidence"}
                        </div>
                    </div>
                </div>
                <div class="conf-gauge-container">
                    <div class="conf-gauge-fill" style="width: {min(100.0, max(5.0, float(conf_pct)))}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Dual Visual Inspection (Original Leaf & Grad-CAM Heatmap)
        gradcam_b64 = res.get("gradcam_base64")
        if gradcam_b64:
            v_col1, v_col2 = st.columns(2, gap="large")
            with v_col1:
                st.markdown("#### 📸 " + ("अपलोड केलेले पान" if is_marathi else "Original Uploaded Leaf"))
                if "last_image" in st.session_state and st.session_state["last_image"]:
                    st.image(st.session_state["last_image"], use_container_width=True)
            with v_col2:
                st.markdown("#### 🔥 " + ("Grad-CAM दृश्य उष्णतानकाशा (Heatmap)" if is_marathi else "Grad-CAM Saliency Heatmap"))
                cam_img = decode_base64_image(gradcam_b64)
                if cam_img:
                    st.image(cam_img, use_container_width=True)
                st.caption(
                    "लाल व पिवळे भाग (Warm highlights) हे रोग ओळखण्यासाठी AI च्या मॉडेलने सर्वाधिक लक्ष दिलेले भाग आहेत."
                    if is_marathi
                    else "Highlighted warm regions indicate foliar symptoms that influenced the MobileNetV3 AI prediction."
                )

        # 3. Audio Readout (TTS)
        tts_text = direct_ans or explanation_text[:250]
        if tts_text:
            st.markdown(
                """
                <div style="margin-top: 1rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.4rem;">🔊</span>
                    <strong style="font-size: 1.05rem; color: #1e293b;">Listen to Full Advisory (ऑडिओ सल्ला ऐका):</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ok, audio_bytes, mime = services["voice"].text_to_speech_bytes(tts_text, lang_code)
            if ok and audio_bytes:
                st.audio(audio_bytes, format=mime)

        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
        st.markdown("### 📋 " + ("तपशीलवार कृषी सल्लागार अहवाल (Detailed Advisory)" if is_marathi else "Comprehensive Agronomic Advisory"))

        # SECTION 1: Integrated Summary
        if direct_ans:
            st.markdown(
                f"""
                <div class="guidance-box summary">
                    <div class="guidance-box-header" style="color: #4338ca;">
                        <span>📋</span> <span>1. {"एकात्मिक कृषी सारांश (Integrated Summary)" if is_marathi else "Integrated Summary & Immediate Action"}</span>
                    </div>
                    <div style="color: #1e293b; line-height: 1.6; font-size: 0.98rem;">
                        {direct_ans}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION 2: Disease Diagnostic Pathology & Symptoms
        if explanation_text:
            st.markdown(
                f"""
                <div class="guidance-box pathology">
                    <div class="guidance-box-header" style="color: #0369a1;">
                        <span>🔬</span> <span>2. {"रोग लक्षणे व विकृतीशास्त्र (Diagnostic Pathology)" if is_marathi else "Diagnostic Pathology & Foliar Symptoms"}</span>
                    </div>
                    <div style="color: #334155; line-height: 1.6; font-size: 0.98rem;">
                        {explanation_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION 3 & 4: Treatment (Curative) & Prevention (Grid Layout)
        t_col1, t_col2 = st.columns(2, gap="medium")

        with t_col1:
            st.markdown(
                f"""
                <div class="guidance-box treatment">
                    <div class="guidance-box-header" style="color: #047857;">
                        <span>💊</span> <span>3. {"उपचार व नियंत्रण पद्धती (Curative Treatment & Dosage)" if is_marathi else "Curative Treatment & Chemical / Organic Management"}</span>
                    </div>
                    <div style="color: #334155; line-height: 1.6; font-size: 0.98rem;">
                        {mgmt_text or 'Apply university-recommended fungicide/bactericide with sticker.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with t_col2:
            st.markdown(
                f"""
                <div class="guidance-box prevention">
                    <div class="guidance-box-header" style="color: #0f766e;">
                        <span>🛡️</span> <span>4. {"प्रतिबंधात्मक उपाय (Preventive & Cultural Practices)" if is_marathi else "Preventive & Cultural Hygiene Practices"}</span>
                    </div>
                    <div style="color: #334155; line-height: 1.6; font-size: 0.98rem;">
                        {prev_text or 'Maintain wide spacing, drip irrigation, and remove infected plant debris.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION 5: Safety Precautions & PHI
        if prec_text:
            st.markdown(
                f"""
                <div class="guidance-box precautions">
                    <div class="guidance-box-header" style="color: #b45309;">
                        <span>⚠️</span> <span>5. {"सुरक्षितता, सावधानता व प्रतीक्षा काळ (Safety Precautions & PHI)" if is_marathi else "Safety Precautions, PPE & Pre-Harvest Interval (PHI)"}</span>
                    </div>
                    <div style="color: #78350f; line-height: 1.6; font-size: 0.98rem;">
                        {prec_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION 6 & 7: Environmental Drivers (Weather + Soil)
        env_col1, env_col2 = st.columns(2, gap="medium")

        with env_col1:
            w_res = res.get("weather") or {}
            st.markdown(
                f"""
                <div class="guidance-box weather-impact">
                    <div class="guidance-box-header" style="color: #1d4ed8;">
                        <span>⛅</span> <span>6. {"स्थानिक हवामान प्रभाव (Microclimate Impact)" if is_marathi else "Local Microclimate Risk & Weather Impact"}</span>
                    </div>
                    <div style="margin-bottom: 0.6rem; font-weight: 600; color: #1e3a8a;">
                        📍 {location_input}: {w_res.get('temperature_c', '--')}°C | {w_res.get('humidity_percent', '--')}% Humidity | Rain Chance: {w_res.get('precipitation_prob_percent', '--')}%
                    </div>
                    <div style="color: #1e293b; line-height: 1.5; font-size: 0.95rem;">
                        {weather_interp or w_res.get('risk_analysis', 'Humid conditions accelerate spore germination.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with env_col2:
            s_res = res.get("soil") or {}
            st.markdown(
                f"""
                <div class="guidance-box soil-impact">
                    <div class="guidance-box-header" style="color: #b45309;">
                        <span>🪴</span> <span>7. {"माती आरोग्य व पोषण प्रभाव (Soil Health Impact)" if is_marathi else "Soil Health & Nutrient Profile"}</span>
                    </div>
                    <div style="margin-bottom: 0.6rem; font-weight: 600; color: #78350f;">
                        🌱 {s_res.get('soil_type', 'Medium Black Soil')} | pH: {s_res.get('pH', 7.2)} | NPK: {s_res.get('nitrogen_kg_ha', 240)}-{s_res.get('phosphorus_kg_ha', 18)}-{s_res.get('potassium_kg_ha', 320)} kg/ha
                    </div>
                    <div style="color: #78350f; line-height: 1.5; font-size: 0.95rem;">
                        {soil_interp or 'Balanced potassium fertilization enhances plant cell wall defense against blight.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # SECTION 8: Research Sources (RAG Citations)
        sources = res.get("sources") or []
        if sources:
            with st.expander("📚 " + ("संशोधन पुरावे व संदर्भ (RAG Research Citations)" if is_marathi else "Research Citations & Scientific Sources (RAG)"), expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {src.get('source_title', 'Agricultural Extension Research')}**")
                    st.write(src.get("content", ""))
                    st.caption(f"Relevance Score: {round(src.get('similarity', 0.85) * 100, 1)}%")


# =============================================================================
# TAB 2: Ask Agronomist (Follow-up & Direct Q&A)
# =============================================================================
with tab_ask:
    st.markdown(
        "### 👨‍🌾 "
        + (
            "कृषी तज्ज्ञांना थेट प्रश्न विचारा (Ask Agricultural Expert)"
            if is_marathi
            else "Ask Agricultural Expert (Direct Q&A)"
        )
    )
    st.caption(
        "फोटो न देता पिकांचे रोग, खत वेळापत्रक, कीड व्यवस्थापन किंवा हवामान जोखीम बद्दल विचारा."
        if is_marathi
        else "Ask any agronomic question on crop health, spray schedules, fertilizers, or pest management without needing a photo."
    )

    q_col1, q_col2 = st.columns([1, 2], gap="large")

    with q_col1:
        st.markdown("#### ⚙️ " + ("संदर्भ निवडा" if is_marathi else "Context Options"))
        crop_context = st.selectbox(
            "🌱 Select Crop Category:" if not is_marathi else "🌱 पिकाचा प्रकार निवडा:",
            options=["Tomato", "Potato", "Bell pepper", "General / All Crops"],
            index=0,
        )
        disease_context = st.text_input(
            "Suspected Disease (Optional):" if not is_marathi else "संशयित रोग (ऐच्छिक):",
            placeholder="e.g. Early blight, Late blight, Leaf spot",
        )

        st.markdown(
            "**" + ("💡 वारंवार विचारले जाणारे प्रश्न:" if is_marathi else "💡 Common Questions:") + "**"
        )
        if st.button("🍅 Tomato fertilizer schedule?" if not is_marathi else "🍅 टोमॅटो खत वेळापत्रक?", use_container_width=True):
            st.session_state["preset_q"] = "What is the best fertilizer schedule for tomato before flowering?"
        if st.button("🥔 Potato blight prevention?" if not is_marathi else "🥔 बटाटा करपा कसा रोखावा?", use_container_width=True):
            st.session_state["preset_q"] = "How to prevent potato blight during humid and overcast weather?"
        if st.button("🌿 Organic neem oil spray?" if not is_marathi else "🌿 सेंद्रिय कडुनिंब फवारणी?", use_container_width=True):
            st.session_state["preset_q"] = "Can I use organic neem oil for controlling aphids and thrips?"

    with q_col2:
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Display previous chat messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input(
            "तुमचा शेतीविषयक प्रश्न येथे लिहा..."
            if is_marathi
            else "Type your farming question here..."
        )

        # Check for preset button click
        if "preset_q" in st.session_state and st.session_state["preset_q"]:
            user_query = st.session_state.pop("preset_q")

        if user_query:
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            with st.spinner("Retrieving research evidence and consulting Gemini..."):
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
                
                # Check for direct answer in localized guidance
                ans_g = ans_res.get("guidance") or {}
                if is_marathi:
                    ans_glang = ans_res.get("guidance_mr") or ans_g.get("mr") or ans_g
                else:
                    ans_glang = ans_res.get("guidance_en") or ans_g.get("en") or ans_g

                answer_text = (
                    ans_glang.get("direct_answer")
                    or ans_res.get("direct_answer")
                    or ans_glang.get("explanation")
                    or "Agronomic guidance generated successfully."
                )

            st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.write(answer_text)

                ok, a_bytes, m_type = services["voice"].text_to_speech_bytes(answer_text, lang_code)
                if ok and a_bytes:
                    st.audio(a_bytes, format=m_type)


# =============================================================================
# TAB 3: Weather & Soil Dashboard
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

    w1, w2, w3, w4 = st.columns(4)
    w1.markdown(
        f"""<div class="ks-metric-tile"><div class="ks-metric-title">🌡️ Temperature</div><div class="ks-metric-value">{w_data.get('temperature_c', '--')}°C</div><div class="ks-metric-sub">Conditions: {w_data.get('condition', 'Clear')}</div></div>""",
        unsafe_allow_html=True,
    )
    w2.markdown(
        f"""<div class="ks-metric-tile"><div class="ks-metric-title">💧 Humidity</div><div class="ks-metric-value">{w_data.get('humidity_percent', '--')}%</div><div class="ks-metric-sub">Foliar Wetness Risk</div></div>""",
        unsafe_allow_html=True,
    )
    w3.markdown(
        f"""<div class="ks-metric-tile"><div class="ks-metric-title">🌧️ Precipitation</div><div class="ks-metric-value">{w_data.get('precipitation_prob_percent', '--')}%</div><div class="ks-metric-sub">Rain Probability</div></div>""",
        unsafe_allow_html=True,
    )
    w4.markdown(
        f"""<div class="ks-metric-tile"><div class="ks-metric-title">💨 Wind Speed</div><div class="ks-metric-value">{w_data.get('wind_speed_kmh', '--')} km/h</div><div class="ks-metric-sub">Optimal for Spraying</div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    soil_col1, soil_col2 = st.columns([1, 1], gap="large")

    with soil_col1:
        st.markdown("#### 🪴 " + ("मातीचे रासायनिक व भौतिक गुणधर्म" if is_marathi else "Regional Soil Profile"))
        st.write(f"• **Regional Soil Classification:** {s_data.get('soil_type', 'Medium Black Soil (Vertisol)')}")
        st.write(f"• **Soil pH Value:** {s_data.get('pH', 7.2)} ({'Optimal Neutral' if 6.5 <= s_data.get('pH', 7.2) <= 7.5 else 'Slightly Alkaline'})")
        st.write(f"• **Organic Carbon:** {s_data.get('organic_carbon_pct', 0.58)}% (Medium Rating)")
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

st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)
st.caption(
    "Krishi Sahayyak • MobileNetV3-Large Classifier • FAISS RAG Knowledge Base • Gemini Multi-Turn Guidance • ICAR & MPKV Rahuri Extension Research"
)
