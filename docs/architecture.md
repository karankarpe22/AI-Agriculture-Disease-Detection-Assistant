# ARCHITECTURE.md — System Architecture & Component Design

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Implemented & Operational  
> **Implementation Language**: Python 3.12+ (Backend) & Vanilla HTML5/CSS3/ES6+ (Frontend)  
> **Primary Paradigm**: Decoupled Perception & Multimodal Grounded Synthesis  

---

## 1. System Overview

The **AI Agriculture Assistant** architecture is designed around a fundamental principle: **Perception must be strictly separated from Generative Synthesis**.

Modern Large Multimodal Models (LMMs), when given a raw image of a diseased crop and asked for a diagnosis and chemical dosage, frequently suffer from:
1. **Diagnostic Hallucination**: Misclassifying visually similar pathogens (e.g., Early Blight vs. Septoria Leaf Spot).
2. **Fabricated Agro-Chemical Dosages**: Recommending inappropriate or banned fungicide concentrations that risk crop burn or toxicity.
3. **Environmental Blindness**: Failing to correlate current humidity and rain conditions with disease severity.

To solve this, the system enforces a strict pipeline boundary:
- **Vision Specialist (MobileNetV3-Large)**: Solely responsible for feature extraction, pattern recognition, and calibrated probabilistic classification across 13 foliar disease classes.
- **Agricultural Grounding (RAG Engine)**: Solely responsible for retrieving peer-reviewed ICAR / TNAU package-of-practices for the diagnosed disease.
- **Environmental Context (Weather Service)**: Gathers live local weather metrics (temperature, relative humidity, precipitation).
- **Generative Advisory (Google Gemini)**: Acts as an empathetic, multilingual synthesizer that compiles the diagnosis, weather risk, and ICAR evidence into structured farmer guidance in English or Marathi.

---

## 2. High-Level Architecture Diagram

```text
====================================================================================================
                                    CLIENT / BROWSER TIER
====================================================================================================
 [Farmer UI: frontend/index.html + styles.css + app.js]
   |
   |-- 1. Drag & Drop Leaf Image (JPG/PNG, <= 10MB)
   |-- 2. Location Input (e.g. Pune, Nashik, Kolhapur)
   |-- 3. Farmer Question (Optional Text or Web Speech STT Voice)
   |-- 4. Language Selection Toggle (English / मराठी)
   |-- 5. Grad-CAM Explainability Checkbox
   |
   v HTTP POST /analyze (multipart/form-data)
====================================================================================================
                              APPLICATION / API TIER (FastAPI)
====================================================================================================
 [FastAPI Router: backend/main.py]
   |
   |-- File & MIME Validation (validate_image_file)
   |
   +---> STEP 1: Vision Preprocessing & Quality Assurance
   |     [src/preprocessing/quality_check.py]
   |     - Resolution check (min 128x128)
   |     - Brightness analysis (mean pixel intensity 40 - 230)
   |     - Sharpness check (Laplacian variance >= 50.0)
   |
   +---> STEP 2: ML Perception Layer (PyTorch)
   |     [src/prediction/predict.py -> src/training/model.py]
   |     - Input tensor: ImageNet transforms -> [1, 3, 224, 224]
   |     - Checkpoint: models/mobilenetv3_best.pth (16.29 MB)
   |     - Inference: MobileNetV3-Large forward pass
   |     - Postprocessing: Softmax -> Class, Confidence %, Top-3 Candidates
   |     - Explainability (Optional): Grad-CAM attention heatmap (model.features[-1])
   |
   +---> STEP 3: Environmental Context Layer
   |     [src/weather/weather_service.py]
   |     - Primary: OpenWeatherMap Current Weather API
   |     - Fallback: Open-Meteo REST API (Zero API key required)
   |     - Computes: Temp (°C), Humidity (%), Rain (mm), Wind (km/h), Microclimate Risk
   |
   +---> STEP 4: Agricultural Knowledge Retrieval (RAG)
   |     [src/rag/vector_store.py -> models/rag_chunks.json]
   |     - 62 curated chunks across 13 ICAR markdown guides
   |     - Deterministic metadata & keyword relevance scoring
   |     - Zero extra RAM overhead (prevents OOM on 512MB hosting)
   |     - Retrieves Top-3 evidence chunks (Chemical, Biological, Cultural)
   |
   +---> STEP 5: Generative Synthesis & Advisory Layer
   |     [src/gemini/guidance_service.py]
   |     - Official Google GenAI SDK (gemini-2.5-flash)
   |     - Ingests: Prediction + Weather + ICAR Evidence + Farmer Question
   |     - Outputs: Structured JSON (Explanation, Weather Impact, Treatment, Prevention)
   |     - Offline Fallback: Deterministic ICAR rule-based synthesizer
   |
   +---> STEP 6: Multilingual Speech Accessibility (TTS)
         [src/utils/voice_service.py]
         - gTTS (Google Text-to-Speech) for Marathi ('mr') & English ('en')
         - Cached locally in outputs/audio_cache/ for instant playback
====================================================================================================
```

---

## 3. Tier & Component Breakdown

### 3.1 Frontend Tier (`frontend/`)
- **`index.html`**: Semantic, accessible HTML5 structure with unified header, split diagnosis/guidance panels, file drag-and-drop dropzone, audio controls, and Grad-CAM modal.
- **`styles.css`**: Vanilla CSS3 design system using HSL color tokens, dark/light glassmorphic card elevations, smooth CSS transitions, and responsive mobile breakpoints.
- **`app.js`**: Client-side application controller managing:
  - Dynamic `API_BASE` resolution across local (`http://localhost:8000`) and cloud environments (`https://krishi-sahayyak.onrender.com`).
  - Image preview, validation feedback, and base64 Grad-CAM rendering.
  - Web Speech API speech-to-text (`webkitSpeechRecognition`) for voice input.
  - Streaming audio playback for gTTS MP3 responses.
  - Complete English/Marathi UI translation via the `I18N` dictionary.

### 3.2 Backend API Tier (`backend/main.py`)
- **Framework**: FastAPI running on Uvicorn ASGI server with Pydantic v2 data models.
- **Middleware**: `CORSMiddleware` configured with permissive cross-origin access for seamless development.
- **Static Mounting**: Mounts `frontend/` as static assets while serving `/`, `/styles.css`, and `/app.js` at root for clean single-port deployment.
- **Endpoints**:
  - `GET /`: Serves `frontend/index.html`.
  - `GET /health`: System diagnostics (`HealthResponse`) reporting model readiness, device, class count, RAG status, and weather provider.
  - `GET /weather`: Standalone microclimate weather and disease pressure lookup.
  - `POST /predict`: Standalone vision inference returning crop, disease, confidence, top-3 candidates, and optional Grad-CAM.
  - `POST /ask`: Q&A advisory endpoint combining disease parameters with RAG and Gemini.
  - `POST /analyze`: Complete end-to-end composite pipeline.
  - `POST /tts`: Accessibility endpoint returning streaming `audio/mpeg` for English and Marathi.

### 3.3 Image Quality & Preprocessing Layer (`src/preprocessing/`)
- **`quality_check.py`**: Defends the perception pipeline against out-of-distribution or degraded inputs:
  - **Resolution Check**: Rejects images smaller than $128 	imes 128$ pixels.
  - **Exposure Check**: Rejects underexposed (mean brightness $< 40.0$) or overexposed (mean brightness $> 230.0$) images.
  - **Blur Check**: Computes Laplacian variance on grayscale leaf images; flags images with variance $< 50.0$ as blurry.
- **`transforms.py`**: PyTorch `torchvision.transforms` builder:
  - `train`: Resize(224, 224), RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter(0.15), ImageNet normalization.
  - `test`/`val`: Resize(224, 224), ImageNet normalization.

### 3.4 ML Perception Layer (`src/prediction/`, `src/training/`, `models/`)
- **Model**: `MobileNetV3-Large` initialized with `MobileNet_V3_Large_Weights.DEFAULT`.
- **Classification Head**: Replaced 1000-class linear projection with custom classifier:
  ```python
  Linear(in_features=960, out_features=1280) -> Hardswish -> Dropout(p=0.2) -> Linear(in_features=1280, out_features=13)
  ```
- **Weights**: Loaded from [`models/mobilenetv3_best.pth`](models/mobilenetv3_best.pth) (16.29 MB) with map_location support for CPU and GPU.
- **Confidence Calibration**: Softmax probabilities are checked against a configured threshold ($	au = 0.60$). Predictions with $	ext{confidence} < 0.60$ flag an `is_low_confidence: True` warning banner.
- **Explainability**: Integrated Grad-CAM computes gradients with respect to `model.features[-1]`, generating a Jet-colormap heatmap overlaid on the original leaf image.

### 3.5 Agricultural Knowledge Base & RAG Layer (`src/rag/`, `knowledge_base/`)
- **Corpus**: 13 comprehensive markdown documents in [`knowledge_base/`](knowledge_base/) sourced from ICAR-IIHR, ICAR-IIVR, ICAR-CPRI, and TNAU Agritech compendiums.
- **Chunks**: Pre-parsed into 62 semantic sections in [`models/rag_chunks.json`](models/rag_chunks.json) categorized by Symptoms, Environmental Conditions, Cultural Management, and Chemical Controls.
- **Retrieval Engine**: Employs structured metadata matching and keyword scoring:
  - Crop match weight: $+6.0$
  - Disease token match weight: $+5.0$ per keyword
  - Heading intent preference (Management/Treatment: $+3.0$, Symptoms: $+1.5$)
  - Farmer question token overlap: $+1.5$ per keyword
- **Zero-RAM Design**: Avoids resident transformer embeddings in RAM, enabling 100% reliable execution on free-tier 512MB hosting containers.

### 3.6 Environmental Context Layer (`src/weather/`)
- **`weather_service.py`**:
  - Primary provider: OpenWeatherMap Current Weather REST API.
  - Fallback provider: Open-Meteo API (requires zero API keys, public open access).
  - Evaluates temperature, humidity, and recent precipitation against fungal and bacterial sporulation thresholds.

### 3.7 Generative Synthesis Layer (`src/gemini/`)
- **SDK**: Official `google-genai` SDK using `gemini-2.5-flash`.
- **Prompt Structure**: Ingests strict ground-truth payload:
  - Diagnosed Disease & Confidence
  - Microclimatic Weather & Fungal Risk
  - Retrieved ICAR Treatment & Biological Guidelines
  - Farmer's Specific Query
  - Target Language (English or Marathi)
- **Safety Invariant**: Explicit system prompt instructions forbid prescribing chemical fungicides outside the retrieved ICAR evidence.

---

## 4. API & Data Flow Specifications

### Composite Analysis Flow (`POST /analyze`)

```text
Client                  FastAPI Router          Predictor             Weather Service       RAG Service           Gemini Service
  |                           |                     |                       |                   |                       |
  |--- 1. POST /analyze ----->|                     |                       |                   |                       |
  |    (image, loc, lang)     |-- 2. Validate ----->|                       |                   |                       |
  |                           |-- 3. predict() ---->|                       |                   |                       |
  |                           |<-- 4. result -------|                       |                   |                       |
  |                           |    (crop, dis, conf)|                       |                   |                       |
  |                           |                                             |                   |                       |
  |                           |-- 5. get_weather(loc) --------------------->|                   |                       |
  |                           |<-- 6. weather_data -------------------------|                   |                       |
  |                           |                                                                 |                       |
  |                           |-- 7. retrieve(crop, dis, question) ---------------------------->|                       |
  |                           |<-- 8. evidence (top 3 ICAR chunks) -----------------------------|                       |
  |                           |                                                                                         |
  |                           |-- 9. generate_guidance(crop, dis, weather, evidence, lang) ---------------------------->|
  |                           |<-- 10. structured advisory -------------------------------------------------------------|
  |                           |
  |<-- 11. JSON Response -----|
  |    (diagnosis + guidance) |
```

---

## 5. Error & Fault Resilience Architecture

| Failure Scenario | Component Handling | System Outcome |
| :--- | :--- | :--- |
| **Dark or Blurry Image Uploaded** | `quality_check.py` | Returns `success: False` with descriptive warning; model inference skipped to prevent garbage prediction. |
| **OpenWeatherMap API Rate Limit / Bad Key** | `weather_service.py` | Automatically falls back to Open-Meteo REST API; weather context is preserved. |
| **Both Weather APIs Unreachable** | `backend/main.py` | Degrades gracefully to default seasonal baseline; disease diagnosis proceeds uninterrupted. |
| **Gemini API Down / Quota 429 Exceeded** | `guidance_service.py` | Triggers `_fallback_grounded_guidance`; formats ICAR markdown chunks into structured cards; zero 500 errors. |
| **Cloud Host Low Memory (< 512MB RAM)** | `vector_store.py` | Uses zero-RAM structured lexical scoring; no heavy transformer loaded; zero OOM crashes. |
