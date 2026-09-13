"""Phase 11: FastAPI backend service for AI Agriculture Assistant."""
from __future__ import annotations

import base64
from dotenv import load_dotenv

load_dotenv()
import io
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.gemini.guidance_service import get_gemini_service
from src.prediction.predict import DiseasePredictor
from src.rag.vector_store import get_rag_service
from src.utils.voice_service import get_voice_service
from src.weather.weather_service import get_weather_service

# Initialize FastAPI app
app = FastAPI(
    title="AI Agriculture Assistant API",
    description="Multilingual Crop Disease Detection and Evidence-Grounded Farmer Guidance",
    version="1.0.0",
)

# Enable CORS for local web interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for direct browser viewing
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Singleton services (lazy loaded)
_predictor: DiseasePredictor | None = None


def get_predictor() -> DiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = DiseasePredictor()
    return _predictor


# Security & file constraints
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def validate_image_file(file: UploadFile, content: bytes) -> None:
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


# --- Pydantic Request & Response Models ---
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    num_classes: int
    classes: list[str]
    rag_ready: bool
    gemini_ready: bool
    weather_provider: str


class AskRequest(BaseModel):
    crop: str
    disease: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    question: str | None = None
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
    """Serve frontend stylesheet."""
    css_file = Path("frontend/styles.css")
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="styles.css not found")


@app.get("/app.js")
def serve_script():
    """Serve frontend javascript."""
    js_file = Path("frontend/app.js")
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
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

    return HealthResponse(
        status="healthy",
        model=model_name,
        device=device,
        num_classes=num_classes,
        classes=classes,
        rag_ready=rag_ready,
        gemini_ready=gemini_ready,
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


@app.post("/predict")
async def predict_crop_disease(
    image: UploadFile = File(..., description="Crop leaf photo"),
    include_gradcam: bool = Form(False),
):
    """Run Image Quality Check -> MobileNetV3 disease classification."""
    content = await image.read()
    validate_image_file(image, content)

    predictor = get_predictor()
    result = predictor.predict(content)

    if include_gradcam and result.get("success"):
        try:
            cam_img = predictor.generate_gradcam(content)
            if cam_img:
                buf = io.BytesIO()
                cam_img.save(buf, format="JPEG")
                result["gradcam_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            result["gradcam_error"] = str(e)

    return result


@app.post("/ask")
def ask_question(request: AskRequest):
    """Retrieve agricultural evidence and generate Gemini guidance."""
    weather_service = get_weather_service()
    weather_data = weather_service.get_weather(city=request.location)

    rag = get_rag_service()
    evidence = rag.retrieve(
        crop=request.crop,
        disease=request.disease,
        question=request.question,
        top_k=3,
    )
    sources = rag.get_sources_summary(evidence)

    gemini = get_gemini_service()
    guidance = gemini.generate_guidance(
        crop=request.crop,
        disease=request.disease,
        confidence=request.confidence,
        weather=weather_data,
        retrieved_evidence=evidence,
        farmer_question=request.question,
        language=request.language,
        is_low_confidence=request.is_low_confidence,
    )

    return {
        "crop": request.crop,
        "disease": request.disease,
        "confidence": request.confidence,
        "weather": weather_data,
        "evidence": evidence,
        "sources": sources,
        "guidance": guidance,
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

    # 1. Image Quality & Classification
    predictor = get_predictor()
    pred_result = predictor.predict(content)

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
        weather_data = weather_service.get_weather(city=location)
    except Exception as e:
        print(f"Weather service error: {e}")
        weather_data = {
            "success": False,
            "provider": "Fallback",
            "temperature_c": 25.0,
            "humidity_percentage": 60,
            "condition": "Normal",
            "risk_analysis": "Weather service currently unavailable; follow standard preventative measures.",
        }

    # 3. RAG Retrieval from Knowledge Base
    try:
        rag = get_rag_service()
        evidence = rag.retrieve(crop=crop, disease=disease, question=question, top_k=3)
        sources = rag.get_sources_summary(evidence)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        evidence = []
        sources = []

    # 4. Contextual Guidance from Gemini
    try:
        gemini = get_gemini_service()
        guidance = gemini.generate_guidance(
            crop=crop,
            disease=disease,
            confidence=confidence,
            weather=weather_data,
            retrieved_evidence=evidence,
            farmer_question=question,
            language=language,
            is_low_confidence=is_low_confidence,
        )
    except Exception as e:
        print(f"Gemini service error: {e}")
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

    # 5. Optional Grad-CAM explainability
    gradcam_base64 = None
    if include_gradcam:
        try:
            cam_img = predictor.generate_gradcam(content)
            if cam_img:
                buf = io.BytesIO()
                cam_img.save(buf, format="JPEG")
                gradcam_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    return {
        "success": True,
        "quality": pred_result.get("quality"),
        "crop": crop,
        "disease": disease,
        "raw_class": pred_result.get("raw_class"),
        "confidence": confidence,
        "confidence_percentage": pred_result.get("confidence_percentage"),
        "is_low_confidence": is_low_confidence,
        "warning": pred_result.get("warning"),
        "top_predictions": pred_result.get("top_predictions", []),
        "weather": weather_data,
        "evidence": evidence,
        "sources": sources,
        "guidance": guidance,
        "gradcam_base64": gradcam_base64,
    }


@app.post("/tts")
def text_to_speech(
    text: str = Form(..., description="Text to speak"),
    language: str = Form("english", description="Language ('english' or 'marathi')"),
):
    """Generate spoken MP3 audio for accessibility in English or Marathi."""
    voice = get_voice_service()
    success, audio_bytes, mime_type = voice.text_to_speech_bytes(text, language)
    if not success:
        raise HTTPException(status_code=500, detail=mime_type)
    return Response(content=audio_bytes, media_type=mime_type)
