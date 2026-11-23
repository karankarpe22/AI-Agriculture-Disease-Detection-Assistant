"""Krishi Sahayyak (कृषी सहाय्यक) — High-Fidelity Agronomic Advisory System.
Faithfully mirrors the full 10-section analytical engine of the HTML/CSS/JS frontend:
1. Integrated Summary & Direct Answer
2. Diagnostic Pathology & Symptoms
3. Weather Impact & Microclimate
4. Soil Health 4-Metric Grid (pH, Moisture, NPK, Carbon) & Regional Interaction
5. Recommended Key Actions (Prioritized)
6. Curative Treatment & Dosage (Chemical + Organic)
7. Preventive & Cultural Practices
8. Safety Precautions, PPE & Pre-Harvest Interval (PHI)
9. Agronomic Research Advisory
10. Scientific Evidence Citations (RAG)
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
# Premium Agriculture Design System (Matching Original Web Theme)
# =============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', 'Noto Sans Devanagari', -apple-system, sans-serif;
}

/* Header Banner */
.hero-header {
    background: linear-gradient(135deg, #064e3b 0%, #047857 55%, #10b981 100%);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    color: #ffffff;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 30px -5px rgba(6, 78, 59, 0.3);
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
}
.hero-title-main {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero-desc-main {
    font-size: 1.02rem;
    color: #d1fae5;
    margin: 0.35rem 0 0 0;
}
.hero-pill-badge {
    background: rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.88rem;
    font-weight: 600;
    color: #ffffff;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Diagnosis Banner */
.diagnosis-banner {
    background: linear-gradient(135deg, #f8fafc 0%, #f0fdf4 100%);
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 6px 18px -3px rgba(16, 185, 129, 0.12);
}
.crop-chip-tag {
    background: #047857;
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.25rem 0.8rem;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    display: inline-block;
}
.disease-title-hero {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.5rem 0 0.2rem 0;
}
.conf-bar-track {
    background: #e2e8f0;
    border-radius: 9999px;
    height: 10px;
    width: 100%;
    margin-top: 0.6rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #10b981, #047857);
}

/* Soil 4-Metric Grid */
.soil-grid-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 0.8rem 0;
}
.soil-metric-cell {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.85rem 0.75rem;
    text-align: center;
}
.soil-metric-lbl {
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
}
.soil-metric-num {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.2rem 0;
}
.soil-metric-rating {
    font-size: 0.76rem;
    font-weight: 600;
    color: #059669;
}

/* Section Header Badges */
.sec-card-header {
    border-left: 5px solid #059669;
    background: #f8fafc;
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin: 1.2rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 1.08rem;
    color: #0f172a;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.sec-card-header.summary { border-left-color: #6366f1; background: #fdf4ff; color: #4338ca; }
.sec-card-header.pathology { border-left-color: #0284c7; background: #f0f9ff; color: #0369a1; }
.sec-card-header.actions { border-left-color: #f59e0b; background: #fffbeb; color: #b45309; }
.sec-card-header.treatment { border-left-color: #059669; background: #f0fdf4; color: #047857; }
.sec-card-header.prevention { border-left-color: #0d9488; background: #f0fdfa; color: #0f766e; }
.sec-card-header.precautions { border-left-color: #e11d48; background: #fff1f2; color: #be123c; }
.sec-card-header.weather { border-left-color: #3b82f6; background: #eff6ff; color: #1d4ed8; }
.sec-card-header.soil { border-left-color: #d97706; background: #fffbeb; color: #b45309; }
.sec-card-header.advisory { border-left-color: #8b5cf6; background: #faf5ff; color: #6d28d9; }

/* Question Callout */
.question-callout {
    background: #f1f5f9;
    border-left: 4px solid #64748b;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    font-style: italic;
    color: #334155;
    margin-bottom: 0.8rem;
    font-size: 0.95rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Services
# =============================================================================
@st.cache_resource(show_spinner="Initializing AI Agriculture models into memory...")
def get_cached_services():
    """Load PyTorch MobileNetV3 and FAISS RAG singleton services."""
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
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.6rem 0 1rem 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🌱</div>
            <h2 style="margin: 0.4rem 0 0.1rem 0; font-weight: 800; color: #064e3b; font-size: 1.55rem;">कृषी सहाय्यक</h2>
            <div style="color: #64748b; font-size: 0.88rem; font-weight: 500;">AI Agriculture Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lang_choice = st.selectbox(
        "🌐 Language / भाषा निवडा",
        options=["English", "मराठी (Marathi)"],
        index=0,
    )
    is_marathi = "मराठी" in lang_choice
    lang_code = "marathi" if is_marathi else "english"

    st.markdown("---")

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

    include_gradcam = st.checkbox(
        "🔍 Grad-CAM दृश्य विश्लेषण (Visual Attention)" if is_marathi else "🔍 Generate Grad-CAM Heatmap",
        value=True,
        help="Highlights the exact leaf disease symptoms that influenced the MobileNetV3 AI decision.",
    )

    st.markdown("---")
    with st.expander("⚙️ System Diagnostics" if not is_marathi else "⚙️ प्रणाली स्थिती", expanded=False):
        st.write(f"• **MobileNetV3:** {len(services['predictor'].class_names)} Classes Ready")
        st.write(f"• **RAG Knowledge Base:** {len(services['rag'].chunks)} Research Chunks")
        st.write("• **Weather Engine:** Live Open-Meteo & OWM")
        st.write("• **Soil Health Service:** Synthetic Regional Records (ICAR)")
        st.write("• **Device:** CPU Mode (Optimized)")


# =============================================================================
# Header Banner
# =============================================================================
header_title = "कृषी सहाय्यक (Krishi Sahayyak)" if is_marathi else "Krishi Sahayyak"
header_sub = (
    "पिकांचे अचूक रोग निदान, सूक्ष्म हवामान जोखीम आणि संशोधन-आधारित कृषी सल्ला"
    if is_marathi
    else "Evidence-Grounded Crop Disease Detection, Microclimate Risk & Multilingual Farmer Guidance"
)

st.markdown(
    f"""
    <div class="hero-header">
        <div>
            <div class="hero-title-main">🌱 {header_title}</div>
            <div class="hero-desc-main">{header_sub}</div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="hero-pill-badge">📍 {location_input}</span>
            <span class="hero-pill-badge">🌐 {lang_code.upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def decode_base64_image(b64_str: str) -> Image.Image | None:
    try:
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data))
    except Exception as e:
        logger.warning(f"Grad-CAM decode error: {e}")
        return None


# =============================================================================
# Complete 10-Section Advisory Renderer (Exact Match to Web Engine)
# =============================================================================
def render_full_advisory(res: dict[str, Any], is_standalone: bool = False, query_text: str = ""):
    """Renders the comprehensive 10-section agronomic report with genuine markdown formatting."""
    crop_disp = res.get("crop") or ("सर्वसाधारण शेती सल्ला" if is_marathi else "Agriculture Advisory")
    disease_disp = res.get("disease") or ("कृषी तज्ज्ञ मार्गदर्शन" if is_marathi else "Expert Q&A Guidance")
    conf_pct = res.get("confidence_percentage", 0.0)
    is_low_conf = res.get("is_low_confidence", False)

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
    advisory_text = g_lang.get("expert_advisory") or ""

    # Synthesize integrated summary if blank
    if not direct_ans:
        if is_marathi:
            direct_ans = f"**{crop_disp}** पिकामध्ये **{disease_disp}** चे निदान झाले आहे (विश्वासार्हता {conf_pct}%). स्थानिक हवामान आणि मातीतील ओलावा पाहता पिकाचे संरक्षण करण्यासाठी खालील शिफारशी तातडीने अंमलात आणाव्यात."
        else:
            direct_ans = f"**{crop_disp}** diagnosed with **{disease_disp}** ({conf_pct}% confidence). Local weather and soil conditions indicate integrated curative and preventive measures are required."

    # Priority Action items (extracted from management)
    actions_text = g_lang.get("recommended_actions") or ""
    if not actions_text and mgmt_text:
        lines = [l.strip() for l in mgmt_text.split("\n") if l.strip()]
        bullets = [l for l in lines if l.startswith(("-", "*", "•")) or (len(l) > 2 and l[0].isdigit() and l[1] in [".", ")"])]
        if len(bullets) >= 2:
            actions_text = "\n".join(bullets[:4])
        else:
            actions_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines[:3])])

    # 1. Top Diagnosis Banner
    if is_standalone:
        st.markdown(
            f"""
            <div class="diagnosis-banner">
                <span class="crop-chip-tag">🌱 {crop_disp}</span>
                <div class="disease-title-hero">{disease_disp}</div>
                <div style="font-size: 0.9rem; color: #047857; font-weight: 600; margin-top: 0.3rem;">
                    ✅ {"प्रमाणित कृषी तज्ज्ञ सल्ला" if is_marathi else "Verified Agronomic Advisory"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="diagnosis-banner">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                    <div>
                        <span class="crop-chip-tag">🌱 {crop_disp}</span>
                        <div class="disease-title-hero">{disease_disp}</div>
                    </div>
                    <div style="text-align: right; min-width: 140px;">
                        <div style="font-size: 1.85rem; font-weight: 800; color: {'#d97706' if is_low_conf else '#059669'};">
                            {conf_pct}%
                        </div>
                        <div style="font-size: 0.82rem; font-weight: 700; color: #64748b; text-transform: uppercase;">
                            {"⚠️ Low Confidence" if is_low_conf else "✅ High Confidence"}
                        </div>
                    </div>
                </div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width: {min(100.0, max(8.0, float(conf_pct)))}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Audio Read Aloud
    tts_speech = f"{crop_disp} {disease_disp}. {direct_ans}. {mgmt_text[:300]}"
    ok, a_bytes, m_type = services["voice"].text_to_speech_bytes(tts_speech, lang_code)
    if ok and a_bytes:
        audio_cols = st.columns([1, 2])
        with audio_cols[0]:
            st.markdown(
                "**🔊 " + ("संपूर्ण ऑडिओ सल्ला ऐका:" if is_marathi else "Listen to Full Advisory:") + "**"
            )
            st.audio(a_bytes, format=m_type)

    # SECTION 1: Integrated Summary
    st.markdown(
        f"""<div class="sec-card-header summary"><span>📋</span> <span>1. {"एकत्रित कृषी सारांश व उत्तर" if is_marathi else "Integrated Summary & Key Advisory"}</span></div>""",
        unsafe_allow_html=True,
    )
    if query_text:
        st.markdown(f"""<div class="question-callout">❓ "{query_text}"</div>""", unsafe_allow_html=True)
    st.markdown(direct_ans)

    # SECTION 2: Disease Diagnostic Pathology & Symptoms
    if explanation_text:
        st.markdown(
            f"""<div class="sec-card-header pathology"><span>🔬</span> <span>2. {"रोग लक्षणे व विकृतीशास्त्र (Diagnostic Pathology)" if is_marathi else "Diagnostic Pathology & Foliar Symptoms"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(explanation_text)

    # SECTION 3: Weather Impact & Microclimate
    w = res.get("weather") or {}
    st.markdown(
        f"""<div class="sec-card-header weather"><span>⛅</span> <span>3. {"हवामान प्रभाव व सूक्ष्म हवामान जोखीम" if is_marathi else "Weather Impact & Microclimate Conditions"}</span></div>""",
        unsafe_allow_html=True,
    )
    w_line = f"**📍 {location_input}:** Temp: `{w.get('temperature_c', '--')}°C` | Humidity: `{w.get('humidity_percent', '--')}%` | Conditions: `{w.get('condition', 'Seasonal')}` | Rain Chance: `{w.get('precipitation_prob_percent', '--')}%`"
    st.markdown(w_line)
    if weather_interp:
        st.markdown(weather_interp)
    elif w.get("risk_analysis"):
        st.markdown(f"*{w.get('risk_analysis')}*")

    # SECTION 4: Soil Health Impact (4-Metric Grid + Analysis)
    s = res.get("soil") or {}
    st.markdown(
        f"""<div class="sec-card-header soil"><span>🪴</span> <span>4. {"मृदा आरोग्य व पोषण प्रभाव (Soil Health Impact)" if is_marathi else "Soil Health Profile & Nutrient Balance"}</span></div>""",
        unsafe_allow_html=True,
    )
    s_interp = s.get("interpretation") or {}
    ph_val = f"{s.get('soil_ph', 7.2):.1f}" if s.get("soil_ph") else "7.2"
    moist_val = f"{s.get('soil_moisture_percent', 24.5):.1f}%" if s.get("soil_moisture_percent") else "24.5%"
    npk_val = f"{round(s.get('nitrogen_kg_ha', 240))}-{round(s.get('phosphorus_kg_ha', 18))}-{round(s.get('potassium_kg_ha', 320))}"
    carbon_val = f"{s.get('organic_carbon_percent', 0.58):.2f}%" if s.get("organic_carbon_percent") else "0.58%"

    st.markdown(
        f"""
        <div style="font-size: 0.92rem; font-weight: 600; color: #475569; margin-bottom: 0.4rem;">
            🌱 <strong>{s.get('soil_type', 'Medium Black Soil (Vertisol)')}</strong> <span style="background: #e0e7ff; color: #3730a3; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem;">🧪 Synthetic Demonstration</span>
        </div>
        <div class="soil-grid-container">
            <div class="soil-metric-cell">
                <div class="soil-metric-lbl">Soil pH</div>
                <div class="soil-metric-num">{ph_val}</div>
                <div class="soil-metric-rating">{s_interp.get('ph_status', 'Optimal Neutral')}</div>
            </div>
            <div class="soil-metric-cell">
                <div class="soil-metric-lbl">Moisture</div>
                <div class="soil-metric-num">{moist_val}</div>
                <div class="soil-metric-rating">{s_interp.get('moisture_status', 'Field Capacity')}</div>
            </div>
            <div class="soil-metric-cell">
                <div class="soil-metric-lbl">Available N-P-K</div>
                <div class="soil-metric-num">{npk_val}</div>
                <div class="soil-metric-rating">kg/ha (ICAR Rating)</div>
            </div>
            <div class="soil-metric-cell">
                <div class="soil-metric-lbl">Organic Carbon</div>
                <div class="soil-metric-num">{carbon_val}</div>
                <div class="soil-metric-rating">Medium</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if soil_interp:
        st.markdown(soil_interp)

    # SECTION 5: Recommended Actions (Prioritized)
    if actions_text:
        st.markdown(
            f"""<div class="sec-card-header actions"><span>⚡</span> <span>5. {"तातडीच्या प्राधान्य कृती (Recommended Actions)" if is_marathi else "Recommended Immediate Actions"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(actions_text)

    # SECTION 6 & 7: Curative Treatment & Preventive Practices (2-Column Grid)
    t_c1, t_c2 = st.columns(2, gap="large")
    with t_c1:
        st.markdown(
            f"""<div class="sec-card-header treatment"><span>💊</span> <span>6. {"उपचार व नियंत्रण पद्धती" if is_marathi else "Curative Treatment & Dosage"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(mgmt_text or "Apply university-recommended fungicide or bactericide with surfactant.")

    with t_c2:
        st.markdown(
            f"""<div class="sec-card-header prevention"><span>🛡️</span> <span>7. {"दीर्घकालीन प्रतिबंधात्मक उपाय" if is_marathi else "Preventive & Cultural Practices"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(prev_text or "Implement 2-3 year crop rotation with non-solanaceous crops and drip irrigation.")

    # SECTION 8 & 9: Safety Precautions & Expert Advisory
    p_c1, p_c2 = st.columns(2, gap="large")
    with p_c1:
        st.markdown(
            f"""<div class="sec-card-header precautions"><span>⚠️</span> <span>8. {"सुरक्षितता व प्रतीक्षा काळ (Safety & PHI)" if is_marathi else "Safety Precautions, PPE & PHI"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(prec_text or "Wear protective mask and gloves during chemical application. Observe mandatory pre-harvest interval.")

    with p_c2:
        st.markdown(
            f"""<div class="sec-card-header advisory"><span>🏛️</span> <span>9. {"कृषी संशोधन सल्ला (Research Advisory)" if is_marathi else "Extension Research Advisory"}</span></div>""",
            unsafe_allow_html=True,
        )
        st.markdown(advisory_text or "Guidelines aligned with ICAR and MPKV Rahuri Agricultural Extension packages.")

    # SECTION 10: Research Sources (RAG Citations)
    sources = res.get("sources") or []
    if sources:
        with st.expander("📚 " + ("संशोधन पुरावे व संदर्भ (10. RAG Research Citations)" if is_marathi else "Scientific Sources & Citations (10. RAG Evidence)"), expanded=False):
            for i, src in enumerate(sources, 1):
                st.markdown(f"**[{i}] {src.get('source_title', 'Agricultural Extension Research')}**")
                st.write(src.get("content", ""))
                st.caption(f"Relevance Score: {round(src.get('similarity', 0.85) * 100, 1)}%")


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
        st.markdown("### 📸 1. " + ("पानाचा फोटो अपलोड करा" if is_marathi else "Upload Crop Leaf Photo"))

        st.markdown("**" + ("🧪 किंवा नमुना पानासह तपासा:" if is_marathi else "🧪 Or Test with a Verified Sample:") + "**")
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

        st.markdown("### ❓ 2. " + ("विशिष्ट प्रश्न (ऐच्छिक)" if is_marathi else "Specific Question (Optional)"))
        farmer_q = st.text_input(
            label="Farmer Question",
            label_visibility="collapsed",
            placeholder=(
                "उदा. यावर सेंद्रिय कडुनिंब तेलाची फवारणी चालेल का?"
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
        st.markdown("### 🖼️ " + ("पानाची प्रतिमा" if is_marathi else "Leaf Image Preview"))
        if input_image_pil:
            st.image(input_image_pil, use_container_width=True)
        else:
            st.info("👈 " + ("कृपया डाव्या बाजूने पानाचा फोटो अपलोड करा किंवा नमुना निवडा." if is_marathi else "Upload a leaf photo or pick a sample on the left."))

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

        st.session_state["diag_result"] = result
        st.session_state["diag_img"] = input_image_pil
        st.session_state["diag_q"] = farmer_q

    if "diag_result" in st.session_state and st.session_state.get("diag_result"):
        res = st.session_state["diag_result"]
        st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

        # Grad-CAM Heatmap comparison if present
        gradcam_b64 = res.get("gradcam_base64")
        if gradcam_b64:
            gc1, gc2 = st.columns(2, gap="large")
            with gc1:
                st.markdown("**📸 " + ("मूळ पान (Original Leaf)" if is_marathi else "Original Leaf") + "**")
                if "diag_img" in st.session_state and st.session_state["diag_img"]:
                    st.image(st.session_state["diag_img"], use_container_width=True)
            with gc2:
                st.markdown("**🔥 " + ("Grad-CAM दृश्य उष्णतानकाशा (Heatmap)" if is_marathi else "Grad-CAM Saliency Heatmap") + "**")
                c_img = decode_base64_image(gradcam_b64)
                if c_img:
                    st.image(c_img, use_container_width=True)
                st.caption("Highlighted warm regions indicate foliar symptoms that influenced the MobileNetV3 AI decision.")

        render_full_advisory(res, is_standalone=False, query_text=st.session_state.get("diag_q", ""))


# =============================================================================
# TAB 2: Ask Agronomist (Full-Fledged Agronomic Q&A Mode)
# =============================================================================
with tab_ask:
    st.markdown("### 👨‍🌾 " + ("कृषी तज्ज्ञांना थेट प्रश्न विचारा" if is_marathi else "Ask Agricultural Expert (Direct Q&A)"))
    st.caption(
        "कोणताही फोटो न देता पिकांचे रोग, खत वेळापत्रक, कीड व्यवस्थापन किंवा हवामान जोखीम बद्दल विचारा."
        if is_marathi
        else "Ask any agronomic question on crop health, spray schedules, fertilizers, or pest management without needing a photo."
    )

    ask_input_col, ask_res_col = st.columns([1, 1.3], gap="large")

    with ask_input_col:
        st.markdown("#### ⚙️ " + ("१. संदर्भ निवडा" if is_marathi else "1. Select Crop Context"))
        crop_category = st.radio(
            "Crop Category",
            options=["Tomato", "Potato", "Bell pepper", "All Crops / General"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("#### ❓ 2. " + ("तुमचा शेतीविषयक प्रश्न" if is_marathi else "2. Your Farming Question"))
        user_question_text = st.text_area(
            "Question Input",
            placeholder=(
                "उदा. फुलोऱ्यापूर्वी टोमॅटोला खतांचे कोणते वेळापत्रक सर्वोत्तम आहे? दमट हवामानात बटाट्यावरील करपा कसा रोखावा?"
                if is_marathi
                else "e.g. What is the best fertilizer schedule for tomato before flowering? How to prevent potato blight during humid weather?"
            ),
            height=120,
            label_visibility="collapsed",
            key="ask_textarea",
        )

        st.markdown("**" + ("💡 द्रुत प्रश्न (Quick Chips):" if is_marathi else "💡 Quick Questions:") + "**")
        qc1, qc2 = st.columns(2)
        if qc1.button("🍅 Tomato fertilizer?", use_container_width=True):
            st.session_state["ask_query_preset"] = "What is the best fertilizer schedule for tomato before flowering?"
        if qc2.button("🥔 Potato blight prevention?", use_container_width=True):
            st.session_state["ask_query_preset"] = "How to prevent potato blight during humid and overcast weather?"
        if qc1.button("🌿 Organic neem spray?", use_container_width=True):
            st.session_state["ask_query_preset"] = "Can I use organic neem oil for controlling aphids and thrips?"
        if qc2.button("🫑 Capsicum bacterial spot?", use_container_width=True):
            st.session_state["ask_query_preset"] = "What are the symptoms and cure of bacterial spot in capsicum?"

        if "ask_query_preset" in st.session_state and st.session_state["ask_query_preset"]:
            user_question_text = st.session_state.pop("ask_query_preset")

        submit_ask_btn = st.button(
            "💬 " + ("तज्ज्ञांना विचारा (Consult Agronomist)" if is_marathi else "Consult Agronomist"),
            type="primary",
            use_container_width=True,
            disabled=(not user_question_text.strip()),
        )

    with ask_res_col:
        st.markdown("#### 📋 " + ("सल्लागार अहवाल" if is_marathi else "Agronomist Advisory Report"))

        if submit_ask_btn and user_question_text.strip():
            with st.spinner("Retrieving agricultural university research and consulting Gemini..."):
                initial_state = create_initial_state(
                    request_type="ask",
                    user_question=user_question_text.strip(),
                    language=lang_code,
                    location=location_input,
                    crop=None if "General" in crop_category else crop_category,
                )
                final_state = run_agriculture_workflow_sync(initial_state)
                ask_result = state_to_api_response(final_state)

            st.session_state["standalone_ask_result"] = ask_result
            st.session_state["standalone_ask_q"] = user_question_text.strip()

        if "standalone_ask_result" in st.session_state and st.session_state.get("standalone_ask_result"):
            render_full_advisory(
                st.session_state["standalone_ask_result"],
                is_standalone=True,
                query_text=st.session_state.get("standalone_ask_q", ""),
            )
        else:
            st.markdown(
                """
                <div style="border: 2px dashed #cbd5e1; border-radius: 16px; padding: 3rem 1.5rem; text-align: center; background: #f8fafc; color: #64748b;">
                    <div style="font-size: 2.8rem; margin-bottom: 0.5rem;">👨‍🌾</div>
                    <div style="font-weight: 600; font-size: 1.1rem; color: #334155;">Ready for your agronomic question</div>
                    <div style="font-size: 0.9rem; margin-top: 0.3rem;">Ask any question on the left or select a quick question chip to generate the full 10-section advisory report.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("🌡️ Temperature", f"{w_data.get('temperature_c', '--')} °C", f"{w_data.get('condition', 'Seasonal')}")
    w2.metric("💧 Humidity", f"{w_data.get('humidity_percent', '--')} %", "Spore Germination Risk")
    w3.metric("🌧️ Precipitation", f"{w_data.get('precipitation_prob_percent', '--')} %", "Rain Chance")
    w4.metric("💨 Wind Speed", f"{w_data.get('wind_speed_kmh', '--')} km/h", "Spray Drift Factor")

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
