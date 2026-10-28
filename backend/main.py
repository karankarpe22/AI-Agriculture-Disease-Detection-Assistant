"""High-Performance FastAPI backend service for AI Agriculture Assistant.
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import io
import logging
import os
from pathlib import Path
from typing import Any

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.gemini.guidance_service import get_gemini_service
from src.prediction.predict import DiseasePredictor
from src.rag.vector_store import get_rag_service
from src.utils.voice_service import get_voice_service
from src.soil.soil_service import get_soil_service
from src.weather.weather_service import get_weather_service

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.main")

# Singleton services (lazy-loaded with startup warmup)
_predictor: DiseasePredictor | None = None


def get_predictor() -> DiseasePredictor:
    """Singleton getter for DiseasePredictor to avoid reloading weights per request."""
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Pre-warms ML models and vector indices on startup."""
    logger.info("Initializing AI Agriculture Assistant Backend...")
    try:
        # Pre-warm MobileNetV3 model into memory
        predictor = await run_in_threadpool(get_predictor)
        logger.info(f"MobileNetV3 loaded ({len(predictor.class_names)} classes) on {predictor.device}.")
    except Exception as e:
        logger.warning(f"Predictor warmup note: {e}")

    try:
        # Pre-warm FAISS RAG index and sentence embeddings
        rag = await run_in_threadpool(get_rag_service)
        logger.info(f"FAISS RAG service loaded ({len(rag.chunks)} research chunks).")
    except Exception as e:
        logger.warning(f"RAG service warmup note: {e}")

    try:
        # Pre-warm Weather microclimate service
        weather = await run_in_threadpool(get_weather_service)
        logger.info("Weather microclimate service initialized.")
    except Exception as e:
        logger.warning(f"Weather service warmup note: {e}")

    try:
        # Pre-warm Soil Health service
        soil_srv = await run_in_threadpool(get_soil_service)
        logger.info(f"Soil Health Agent loaded ({len(soil_srv.records)} synthetic regional records).")
    except Exception as e:
        logger.warning(f"Soil service warmup note: {e}")

    yield
    logger.info("Shutting down AI Agriculture Assistant Backend.")


# Initialize FastAPI app with Lifespan
app = FastAPI(
    title="AI Agriculture Assistant API",
    description="Multilingual Crop Disease Detection, Microclimate Risk & Evidence-Grounded Farmer Guidance",
    version="1.0.0",
    lifespan=lifespan,
)

# GZip Compression Middleware (compresses responses > 1000 bytes)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enable CORS strictly for authorized frontend domains and local development
ALLOWED_ORIGINS = [
    "https://krishi-sahayyak.vercel.app",
    "https://krishi-sahayyak.onrender.com",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount frontend directory for direct browser viewing
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Mount data directory for sample leaf images and static datasets
data_dir = Path("frontend/data")
if not data_dir.exists():
    data_dir = Path("data")
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

# Security & file constraints
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image_file(file: UploadFile, content: bytes) -> None:
    """Defensive validation for image upload: file size, extension, and MIME type."""
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of 10MB ({len(content) / (1024*1024):.2f}MB received).",
        )
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type '{file.content_type}'. Must be a valid JPEG, PNG, or WebP image.",
        )



# --- Comprehensive Multilingual Localization Tables ---
CROP_TRANSLATIONS = {
    "Potato": {"en": "Potato", "mr": "बटाटा"},
    "Tomato": {"en": "Tomato", "mr": "टोमॅटो"},
    "Pepper, bell": {"en": "Bell pepper", "mr": "ढोबळी मिरची"},
    "Bell pepper": {"en": "Bell pepper", "mr": "ढोबळी मिरची"},
    "General / Solanaceae": {"en": "General Crop", "mr": "सर्व पिके / शेती सल्ला"},
    "General": {"en": "General Crop", "mr": "सर्व पिके / शेती सल्ला"},
}

DISEASE_TRANSLATIONS = {
    "Early blight": {"en": "Early blight", "mr": "लवकर येणारा करपा (Early Blight)"},
    "Late blight": {"en": "Late blight", "mr": "उशिरा येणारा करपा (Late Blight)"},
    "Bacterial spot": {"en": "Bacterial spot", "mr": "जिवाणूजन्य ठिपके (Bacterial Spot)"},
    "Leaf Mold": {"en": "Leaf Mold", "mr": "पानावरील बुरशी (Leaf Mold)"},
    "Septoria leaf spot": {"en": "Septoria leaf spot", "mr": "सेप्टोरिया पानावरील ठिपके (Septoria Leaf Spot)"},
    "Spider mites Two-spotted spider mite": {"en": "Two-spotted spider mite", "mr": "दोन ठिपक्यांची लाल कोळी (Spider Mites)"},
    "Target Spot": {"en": "Target Spot", "mr": "लक्ष्य ठिपके (Target Spot)"},
    "Tomato Yellow Leaf Curl Virus": {"en": "Yellow Leaf Curl Virus", "mr": "पिवळा चुरडा-मुरडा विषाणू (TYLCV)"},
    "Tomato mosaic virus": {"en": "Tomato mosaic virus", "mr": "टोमॅटो मोझॅक विषाणू (Mosaic Virus)"},
    "healthy": {"en": "Healthy Crop", "mr": "निरोगी पीक (Healthy)"},
    "Healthy": {"en": "Healthy Crop", "mr": "निरोगी पीक (Healthy)"},
    "General Agronomic Inquiry": {"en": "Expert Q&A Guidance", "mr": "कृषी तज्ज्ञ सल्ला व मार्गदर्शन"},
}

CITY_TRANSLATIONS_MR = {
    "pune": "पुणे",
    "akurdi": "आकुर्डी",
    "nashik": "नाशिक",
    "nagpur": "नागपूर",
    "aurangabad": "छत्रपती संभाजीनगर",
    "chhatrapati sambhajinagar": "छत्रपती संभाजीनगर",
    "solapur": "सोलापूर",
    "kolhapur": "कोल्हापूर",
    "satara": "सातारा",
    "sangli": "सांगली",
    "jalgaon": "जळगाव",
    "ahmednagar": "अहमदनगर",
    "amravati": "अमरावती",
    "nanded": "नांदेड",
    "mumbai": "मुंबई",
    "thane": "ठाणे",
    "baramati": "बारामती",
}

WEATHER_CONDITIONS_MR = {
    "clear": "स्वच्छ आकाश",
    "sunny": "सूर्यप्रकाश",
    "partly cloudy": "अंशतः ढगाळ",
    "cloudy": "ढगाळ हवामान",
    "overcast": "पूर्ण ढगाळ",
    "rain": "पाऊस",
    "light rain": "हलका पाऊस",
    "moderate rain": "मध्यम पाऊस",
    "heavy rain": "मुसळधार पाऊस",
    "thunderstorm": "वादळी पाऊस",
    "drizzle": "रिमझिम पाऊस",
    "fog": "धुके",
    "mist": "धुके",
    "haze": "धूसर",
    "normal": "सामान्य हवामान",
    "seasonal": "हंगामी हवामान",
}


def localize_weather_data(weather: dict[str, Any]) -> dict[str, Any]:
    """Enrich weather dictionary with Marathi localization fields."""
    if not isinstance(weather, dict):
        return weather
    w = dict(weather)
    loc = str(w.get("location", "")).lower()
    city_mr = None
    for k, v in CITY_TRANSLATIONS_MR.items():
        if k in loc:
            city_mr = f"{v}, भारत"
            break
    w["location_mr"] = city_mr or f"{w.get('location', 'शेत')}"

    cond = str(w.get("condition", "")).lower()
    w["condition_mr"] = WEATHER_CONDITIONS_MR.get(cond, WEATHER_CONDITIONS_MR.get(cond.strip(), w.get("condition", "हंगामी हवामान")))

    risk = str(w.get("risk_analysis", ""))
    hum = w.get("humidity_percentage", 60)
    if "high humidity" in risk.lower() or "humidity" in risk.lower():
        w["risk_analysis_mr"] = f"हवेतील जास्त आर्द्रता ({hum}%) बुरशीच्या बीजाणूंचा प्रसार वेगाने होण्यास अनुकूल आहे. शेतात पाण्याचा निचरा योग्य ठेवा."
    elif "rain" in risk.lower():
        w["risk_analysis_mr"] = "पावसाची शक्यता असल्याने फवारणी करताना स्टीकर (Sticker) वापरा किंवा पाऊस थांबल्यावर फवारा."
    else:
        w["risk_analysis_mr"] = "सध्याचे स्थानिक हवामान सामान्य असून नियमित निरीक्षण व प्रतिबंधात्मक काळजी ठेवा."
    return w


# --- Pydantic Request & Response Models ---
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    num_classes: int
    classes: list[str]
    rag_ready: bool
    gemini_ready: bool
    soil_ready: bool
    weather_provider: str



class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text content to translate")
    target_lang: str = Field(default="marathi", description="Target language ('marathi' or 'english')")
    crop: str | None = None
    disease: str | None = None


class AskRequest(BaseModel):
    crop: str | None = None
    disease: str | None = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    question: str = Field(..., description="Farmer question")
    location: str | None = "Pune"
    language: str = "english"
    is_low_confidence: bool = False


# --- Endpoints ---

@app.get("/")
def serve_index():
    """Serve frontend index.html."""
    index_file = Path("frontend/index.html")
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Agriculture Assistant API is operational. Visit /docs for Swagger UI."}


@app.get("/styles.css")
def serve_styles():
    """Serve frontend stylesheet with caching header."""
    css_file = Path("frontend/styles.css")
    if css_file.exists():
        return FileResponse(
            css_file,
            media_type="text/css",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
def serve_script():
    """Serve frontend javascript with caching header."""
    js_file = Path("frontend/app.js")
    if js_file.exists():
        return FileResponse(
            js_file,
            media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    raise HTTPException(status_code=404, detail="app.js not found")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Provide a sprout SVG favicon to avoid browser 404 logs."""
    svg_icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🌱</text></svg>'
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check reporting model, RAG, and service readiness."""
    try:
        predictor = get_predictor()
        num_classes = len(predictor.class_names)
        classes = predictor.class_names
        device = str(predictor.device)
        model_name = "MobileNetV3-Large"
    except Exception as e:
        num_classes = 0
        classes = []
        device = "unavailable"
        model_name = f"Error: {e}"

    rag_ready = Path("models/rag_index.faiss").exists() or Path("knowledge_base").exists()
    gemini_ready = bool(os.getenv("GEMINI_API_KEY", "").strip())
    weather_provider = "OpenWeatherMap" if os.getenv("WEATHER_API_KEY", "").strip() else "Open-Meteo (Live)"

    soil_ready = len(get_soil_service().records) > 0
    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device,
        num_classes=num_classes,
        classes=classes,
        rag_ready=rag_ready,
        gemini_ready=gemini_ready,
        soil_ready=soil_ready,
        weather_provider=weather_provider,
    )


@app.get("/weather")
def get_weather(
    city: str | None = Query(None, description="City name (e.g. Pune, Nashik)"),
    lat: float | None = Query(None, description="Latitude coordinate"),
    lon: float | None = Query(None, description="Longitude coordinate"),
):
    """Retrieve real-time weather and crop disease microclimate risk analysis."""
    service = get_weather_service()
    result = service.get_weather(city=city, latitude=lat, longitude=lon)
    return result


@app.get("/soil")
def get_soil_profile(
    city: str | None = Query(None, description="City or farmer location (e.g. Pune, Nashik)"),
    crop: str | None = Query(None, description="Crop category (e.g. Tomato, Potato, Bell pepper)"),
):
    """Retrieve synthetic agro-climatic soil health profile and tripartite interaction interpretation."""
    service = get_soil_service()
    result = service.get_soil_health(location=city, crop=crop)
    return result


@app.post("/predict")
async def predict_crop_disease(
    image: UploadFile = File(..., description="Crop leaf photo"),
    include_gradcam: bool = Form(False),
):
    """Run Image Quality Check -> MobileNetV3 disease classification offloaded to threadpool."""
    content = await image.read()
    validate_image_file(image, content)

    predictor = get_predictor()
    result = await run_in_threadpool(predictor.predict, content)

    if include_gradcam and result.get("success"):
        try:
            cam_img = await run_in_threadpool(predictor.generate_gradcam, content)
            if cam_img:
                buf = io.BytesIO()
                cam_img.save(buf, format="JPEG")
                result["gradcam_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            result["gradcam_error"] = str(e)

    return result


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Retrieve agricultural evidence and generate Gemini guidance for follow-up or standalone queries."""
    weather_service = get_weather_service()
    weather_data = await run_in_threadpool(weather_service.get_weather, city=request.location)

    # Determine crop and disease context (support standalone Q&A without image)
    crop = request.crop
    if not crop and request.question:
        q_lower = request.question.lower()
        if "tomato" in q_lower or "टोमॅटो" in q_lower:
            crop = "Tomato"
        elif "potato" in q_lower or "बटाटा" in q_lower:
            crop = "Potato"
        elif any(k in q_lower for k in ["pepper", "capsicum", "शिमला", "मिरची"]):
            crop = "Bell pepper"

    resolved_crop = crop or "General / Solanaceae"
    resolved_disease = request.disease or "General Agronomic Inquiry"

    rag = get_rag_service()
    evidence = await run_in_threadpool(
        rag.retrieve,
        crop=crop if crop and crop != "General / Solanaceae" else None,
        disease=request.disease,
        question=request.question,
        top_k=3,
    )
    sources = rag.get_sources_summary(evidence)

    soil_service = get_soil_service()
    soil_data = await run_in_threadpool(
        soil_service.get_soil_health,
        location=request.location,
        crop=resolved_crop,
    )

    gemini = get_gemini_service()
    guidance = await run_in_threadpool(
        gemini.generate_guidance,
        crop=resolved_crop,
        disease=resolved_disease,
        confidence=request.confidence,
        weather=weather_data,
        retrieved_evidence=evidence,
        farmer_question=request.question,
        language=request.language,
        is_low_confidence=request.is_low_confidence,
        soil=soil_data,
    )

    crop_en = CROP_TRANSLATIONS.get(resolved_crop, {}).get("en", resolved_crop)
    crop_mr = CROP_TRANSLATIONS.get(resolved_crop, {}).get("mr", resolved_crop)
    disease_en = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("en", resolved_disease)
    disease_mr = DISEASE_TRANSLATIONS.get(resolved_disease, {}).get("mr", resolved_disease)

    guidance_en = guidance.get("en", guidance)
    guidance_mr = guidance.get("mr", guidance)
    direct_answer_en = guidance.get("direct_answer_en") or (guidance_en.get("direct_answer") if isinstance(guidance_en, dict) else "") or guidance.get("direct_answer", "")
    direct_answer_mr = guidance.get("direct_answer_mr") or (guidance_mr.get("direct_answer") if isinstance(guidance_mr, dict) else "") or guidance.get("direct_answer", "")

    localized_weather = localize_weather_data(weather_data)
    is_mr = (request.language or "").lower() in ["marathi", "mr", "मराठी"]
    final_crop = crop_mr if is_mr else crop_en
    final_disease = disease_mr if is_mr else disease_en
    final_direct_answer = direct_answer_mr if is_mr and direct_answer_mr else direct_answer_en

    return {
        "success": True,
        "crop": final_crop,
        "crop_en": crop_en,
        "crop_mr": crop_mr,
        "disease": final_disease,
        "disease_en": disease_en,
        "disease_mr": disease_mr,
        "confidence": request.confidence,
        "question": request.question,
        "direct_answer": final_direct_answer,
        "direct_answer_en": direct_answer_en,
        "direct_answer_mr": direct_answer_mr,
        "weather": localized_weather,
        "soil": soil_data,
        "evidence": evidence,
        "sources": sources,
        "guidance": guidance,
        "guidance_en": guidance_en,
        "guidance_mr": guidance_mr,
    }


@app.post("/analyze")
async def analyze_leaf(
    image: UploadFile = File(..., description="Crop leaf photo"),
    question: str | None = Form(None, description="Farmer question"),
    location: str | None = Form("Pune", description="Farmer location or city"),
    language: str = Form("english", description="Preferred language ('english' or 'marathi')"),
    include_gradcam: bool = Form(False),
):
    """End-to-End Pipeline: Image -> Quality Check -> MobileNetV3 -> Weather -> RAG -> Gemini."""
    content = await image.read()
    validate_image_file(image, content)

    # 1. Image Quality & Classification (offloaded to threadpool)
    predictor = get_predictor()
    pred_result = await run_in_threadpool(predictor.predict, content)

    # If quality check failed, return early with informative warning
    if not pred_result.get("success"):
        return {
            "success": False,
            "error_type": pred_result.get("error_type"),
            "quality": pred_result.get("quality"),
            "warning": pred_result.get("warning"),
            "crop": None,
            "disease": None,
            "confidence": 0.0,
        }

    crop = pred_result["crop"]
    disease = pred_result["disease"]
    confidence = pred_result["confidence"]
    is_low_confidence = pred_result["is_low_confidence"]

    # 2. Live Weather context
    try:
        weather_service = get_weather_service()
        weather_data = await run_in_threadpool(weather_service.get_weather, city=location)
    except Exception as e:
        logger.warning(f"Weather service error: {e}")
        weather_data = {
            "success": False,
            "provider": "Fallback",
            "temperature_c": 25.0,
            "humidity_percentage": 60,
            "condition": "Normal",
            "risk_analysis": "Weather service currently unavailable; follow standard preventative measures.",
        }

    # 3. RAG Retrieval from Knowledge Base (offloaded to threadpool)
    try:
        rag = get_rag_service()
        evidence = await run_in_threadpool(rag.retrieve, crop=crop, disease=disease, question=question, top_k=3)
        sources = rag.get_sources_summary(evidence)
    except Exception as e:
        logger.warning(f"RAG retrieval error: {e}")
        evidence = []
        sources = []

    # 4. Synthetic Soil Health Context (offloaded to threadpool)
    try:
        soil_service = get_soil_service()
        soil_data = await run_in_threadpool(
            soil_service.get_soil_health,
            location=location,
            crop=crop,
        )
    except Exception as e:
        logger.warning(f"Soil service error: {e}")
        soil_data = {
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
        }

    # 5. Contextual Guidance from Gemini with Tripartite Synthesis (offloaded to threadpool)
    try:
        gemini = get_gemini_service()
        guidance = await run_in_threadpool(
            gemini.generate_guidance,
            crop=crop,
            disease=disease,
            confidence=confidence,
            weather=weather_data,
            retrieved_evidence=evidence,
            farmer_question=question,
            language=language,
            is_low_confidence=is_low_confidence,
            soil=soil_data,
        )
    except Exception as e:
        logger.warning(f"Gemini service error: {e}")
        guidance = {
            "explanation": f"Observed foliar symptoms are characteristic of {crop} {disease}.",
            "weather_interpretation": "Maintain regular monitoring under prevailing regional conditions.",
            "management_guidance": "Apply ICAR-recommended cultural practices and consult local agro-dealers for registered fungicides.",
            "prevention": "Prune affected leaves, avoid overhead irrigation, and sanitize tools.",
            "precautions": "Wear personal protective equipment (PPE) when handling chemical solutions.",
            "uncertainty_warning": None,
            "expert_advisory": "Contact your nearest Krishi Vigyan Kendra (KVK) or Block Agriculture Officer for on-field verification.",
            "language": language,
            "powered_by": "ICAR Grounded Advisory",
        }

    # 5. Optional Grad-CAM explainability (offloaded to threadpool)
    gradcam_base64 = None
    if include_gradcam:
        try:
            cam_img = await run_in_threadpool(predictor.generate_gradcam, content)
            if cam_img:
                buf = io.BytesIO()
                cam_img.save(buf, format="JPEG")
                gradcam_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.warning(f"Grad-CAM error: {e}")

    crop_en = CROP_TRANSLATIONS.get(crop, {}).get("en", crop)
    crop_mr = CROP_TRANSLATIONS.get(crop, {}).get("mr", crop)
    disease_en = DISEASE_TRANSLATIONS.get(disease, {}).get("en", disease)
    disease_mr = DISEASE_TRANSLATIONS.get(disease, {}).get("mr", disease)

    guidance_en = guidance.get("en", guidance)
    guidance_mr = guidance.get("mr", guidance)
    direct_answer_en = guidance.get("direct_answer_en") or (guidance_en.get("direct_answer") if isinstance(guidance_en, dict) else "") or guidance.get("direct_answer", "")
    direct_answer_mr = guidance.get("direct_answer_mr") or (guidance_mr.get("direct_answer") if isinstance(guidance_mr, dict) else "") or guidance.get("direct_answer", "")

    localized_weather = localize_weather_data(weather_data)
    is_mr = (language or "").lower() in ["marathi", "mr", "मराठी"]
    final_crop = crop_mr if is_mr else crop_en
    final_disease = disease_mr if is_mr else disease_en
    final_direct_answer = direct_answer_mr if is_mr and direct_answer_mr else direct_answer_en

    return {
        "success": True,
        "quality": pred_result.get("quality"),
        "crop": final_crop,
        "crop_en": crop_en,
        "crop_mr": crop_mr,
        "disease": final_disease,
        "disease_en": disease_en,
        "disease_mr": disease_mr,
        "raw_class": pred_result.get("raw_class"),
        "confidence": confidence,
        "confidence_percentage": pred_result.get("confidence_percentage"),
        "is_low_confidence": is_low_confidence,
        "warning": pred_result.get("warning"),
        "top_predictions": pred_result.get("top_predictions", []),
        "weather": localized_weather,
        "soil": soil_data,
        "evidence": evidence,
        "sources": sources,
        "guidance": guidance,
        "guidance_en": guidance_en,
        "guidance_mr": guidance_mr,
        "direct_answer": final_direct_answer,
        "direct_answer_en": direct_answer_en,
        "direct_answer_mr": direct_answer_mr,
        "farmer_question": question,
        "gradcam_base64": gradcam_base64,
    }



@app.post("/translate")
async def translate_text(request: TranslateRequest):
    """Translate arbitrary agricultural text or questions between English and Marathi."""
    text = (request.text or "").strip()
    target_lang = (request.target_lang or "marathi").lower().strip()
    if not text:
        return {"success": True, "translated_text": ""}

    is_to_marathi = target_lang in ["marathi", "mr", "मराठी"]

    # 1. Quick phrase dictionary lookup for common agronomic follow-ups
    phrase_map_mr = {
        "can i use organic neem oil for this?": "यावर सेंद्रिय कडुनिंब तेलाची फवारणी करता येईल का?",
        "what is the waiting period before harvesting fruits?": "फळे किंवा कंद काढणीपूर्वी किती दिवस प्रतीक्षा करावी (PHI काळ)?",
        "when is the best time of day to spray?": "फवारणी करण्यासाठी दिवसाची कोणती वेळ सर्वोत्तम आहे?",
        "how often should i repeat this spray?": "ही फवारणी किती दिवसांच्या अंतराने पुन्हा करावी?",
        "what care should be taken to prevent this disease": "हा रोग रोखण्यासाठी काय काळजी घ्यावी?",
        "what spray should i use to treat this disease?": "या रोगावर नियंत्रणासाठी कोणती फवारणी करावी?",
        "how can i prevent this in my other plants?": "इतर झाडांवर हा रोग पसरू नये म्हणून काय करावे?",
        "is this disease dangerous for fruit harvest?": "हा रोग फळांच्या उत्पादनासाठी किती घातक आहे?",
    }
    phrase_map_en = {v: k for k, v in phrase_map_mr.items()}

    lower_text = text.lower().strip(" ?.")
    for k, v in (phrase_map_mr if is_to_marathi else phrase_map_en).items():
        if k.strip(" ?.") in lower_text or lower_text in k.strip(" ?."):
            return {"success": True, "translated_text": v, "method": "dictionary"}

    # 2. Try Gemini translation if client is available
    gemini = get_gemini_service()
    client = gemini._get_client()
    if client:
        try:
            target_desc = "authentic spoken Marathi for a farmer in Maharashtra" if is_to_marathi else "clear, simple English"
            prompt = f"Translate the following agricultural query or advice accurately into {target_desc}. Do not add extra chatter, return only the direct translation:\n\n{text}"
            response = await run_in_threadpool(client.models.generate_content, model="gemini-2.5-flash", contents=prompt)
            tr_text = response.text.strip()
            return {"success": True, "translated_text": tr_text, "method": "gemini"}
        except Exception as e:
            logger.warning(f"Gemini translation fallback: {e}")

    # 3. Heuristic translation fallback
    return {"success": True, "translated_text": text, "method": "original"}


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="Text to speak"),
    language: str = Form("english", description="Language ('english' or 'marathi')"),
):
    """Generate spoken MP3 audio for accessibility in English or Marathi offloaded to threadpool."""
    voice = get_voice_service()
    success, audio_bytes, mime_type = await run_in_threadpool(voice.text_to_speech_bytes, text, language)
    if not success:
        raise HTTPException(status_code=500, detail=mime_type)
    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )
