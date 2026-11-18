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

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import torch

from src.gemini.guidance_service import get_gemini_service
from src.prediction.service import get_predictor
from src.rag.vector_store import get_rag_service
from src.utils.memory import log_memory
from src.utils.voice_service import get_voice_service
from src.soil.soil_service import get_soil_service
from src.weather.weather_service import get_weather_service
from src.graph.state import create_initial_state, state_to_api_response
from src.graph.workflow import run_agriculture_workflow

# Keep CPU thread pool compact on low-resource container environments (Render 512MB RAM)
if not torch.cuda.is_available():
    torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("backend.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Pre-warms ML models and vector indices on startup."""
    log_memory("Startup Begin", logger)
    logger.info("Initializing AI Agriculture Assistant Backend...")
    try:
        # Pre-warm shared MobileNetV3 model into memory
        predictor = await run_in_threadpool(get_predictor)
        log_memory("After MobileNetV3 Warmup", logger)
        logger.info(f"MobileNetV3 loaded ({len(predictor.class_names)} classes) on {predictor.device}.")
    except Exception as e:
        logger.warning(f"Predictor warmup note: {e}")

    try:
        # Pre-warm FAISS RAG index and chunks metadata into memory
        rag = await run_in_threadpool(get_rag_service)
        await run_in_threadpool(rag.load_cached_index)
        log_memory("After RAG Index Load", logger)
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

    log_memory("Startup Complete", logger)
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
    if not content or len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes received).",
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds maximum limit of 10MB.",
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



# --- Comprehensive Multilingual Localization Tables & Enrichment ---
from src.utils.localization import (
    CROP_TRANSLATIONS,
    DISEASE_TRANSLATIONS,
    CITY_TRANSLATIONS_MR,
    WEATHER_CONDITIONS_MR,
    localize_weather_data,
)


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

    try:
        rag_service = get_rag_service()
        rag_ready = rag_service.is_ready()
    except Exception:
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
    log_memory("Before /predict", logger)
    content = await image.read(MAX_FILE_SIZE_BYTES + 1)
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

    log_memory("After /predict", logger)
    return result


@app.post("/ask")
async def ask_question(request: AskRequest):
    """Retrieve agricultural evidence and generate Gemini guidance for follow-up or standalone queries."""
    log_memory("Before /ask", logger)
    initial_state = create_initial_state(
        request_type="ask",
        user_question=request.question,
        language=request.language,
        location=request.location,
        crop=request.crop,
        disease=request.disease,
        confidence=request.confidence,
        is_low_confidence=request.is_low_confidence,
    )
    final_state = await run_agriculture_workflow(initial_state)
    log_memory("After /ask", logger)
    return state_to_api_response(final_state)


@app.post("/analyze")
async def analyze_leaf(
    image: UploadFile = File(..., description="Crop leaf photo"),
    question: str | None = Form(None, description="Farmer question"),
    location: str | None = Form("Pune", description="Farmer location or city"),
    language: str = Form("english", description="Preferred language ('english' or 'marathi')"),
    include_gradcam: bool = Form(False),
):
    """End-to-End Pipeline: Image -> Quality Check -> MobileNetV3 -> Weather -> RAG -> Gemini."""
    log_memory("Before /analyze", logger)
    content = await image.read(MAX_FILE_SIZE_BYTES + 1)
    validate_image_file(image, content)

    initial_state = create_initial_state(
        request_type="analyze",
        user_question=question,
        language=language,
        location=location,
        image_input=content,
        include_gradcam=include_gradcam,
    )
    final_state = await run_agriculture_workflow(initial_state)
    log_memory("After /analyze", logger)
    return state_to_api_response(final_state)



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
