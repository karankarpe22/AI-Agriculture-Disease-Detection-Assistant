# AI Agriculture Assistant (कृषी सहाय्यक)

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-FF6F00?logo=chainlink&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Tests Passing](https://img.shields.io/badge/Tests-144%20Passed-brightgreen)](tests/)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel&logoColor=white)](https://krishi-sahayyak.vercel.app)
[![Render](https://img.shields.io/badge/Backend-Render-46E3B7?logo=render&logoColor=white)](https://krishi-sahayyak.onrender.com)

An academically grounded, evidence-based agricultural decision support platform. It integrates a **MobileNetV3-Large** convolutional neural network for foliar disease classification, a **LangGraph** asynchronous stateful orchestration engine, a **FAISS-grounded RAG** knowledge layer from ICAR agronomic guides, **real-time weather epidemiological risk analysis**, an **agro-climatic synthetic soil health agent**, and **Google Gemini** generative advisory in **English and Marathi**, delivered with neural voice synthesis.

---

## 🌐 Live Production Deployments

- **Frontend Application (Vercel):** [https://krishi-sahayyak.vercel.app](https://krishi-sahayyak.vercel.app)
- **Backend API Service (Render):** [https://krishi-sahayyak.onrender.com](https://krishi-sahayyak.onrender.com)
- **Interactive Swagger Documentation:** [https://krishi-sahayyak.onrender.com/docs](https://krishi-sahayyak.onrender.com/docs)
- **Health Check Endpoint:** [https://krishi-sahayyak.onrender.com/health](https://krishi-sahayyak.onrender.com/health)

---

## 1. Project Overview & Problem Statement

Smallholder farmers in India frequently face severe crop yield losses due to delayed or inaccurate diagnosis of foliar diseases. While modern computer vision models can classify leaf pathology, standalone classifiers:
1. **Lack context:** They cannot account for microclimatic weather conditions (e.g., persistent humidity accelerating fungal sporulation).
2. **Lack soil awareness:** They ignore regional soil pH, moisture regimes, and macronutrient balances that influence plant vulnerability.
3. **Risk hallucination:** When generic LLMs are prompted directly with leaf images, they hallucinate hazardous chemical fungicide dosages.
4. **Lack regional accessibility:** Farmers require immediate diagnostic guidance in their native languages (Marathi / English) delivered via voice.

The **AI Agriculture Assistant (Krishi Sahayyak)** resolves this by decoupling perception from generative reasoning using a **LangGraph-orchestrated multi-agent pipeline**:
- **Perception:** A fine-tuned **MobileNetV3-Large** model identifies leaf pathology across 13 classes with calibrated confidence and **Grad-CAM** visual heatmaps.
- **Microclimate Telemetry:** A **Weather Service** queries live OpenWeatherMap with automatic zero-key fallback to Open-Meteo.
- **Soil Health:** An **Agro-Climatic Soil Agent** matches regional profiles against a validated 100-record Maharashtra synthetic soil database using a 5-tier fallback hierarchy.
- **Agronomic Grounding:** A **FAISS Vector Store** retrieves verified management practices from ICAR (Indian Council of Agricultural Research) documents.
- **Generative Synthesis:** **Google Gemini** synthesizes the tripartite context into safe, actionable advisory in English and Marathi, with a deterministic offline RAG template fallback.
- **Neural Voice:** **Microsoft Azure Neural TTS** (via `edge-tts`) and `gTTS` stream authentic regional speech directly to the user.

---

## 2. LangGraph StateGraph Architecture

The core analysis pipeline (`/analyze`) and conversational agronomist mode (`/ask`) are orchestrated through an asynchronous compiled **LangGraph `StateGraph`**:

```text
                                  [START]
                                     │
                             (Input Mode Router)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             [quality_node]                  [setup_ask_context]
                    │                                 │
           ┌────────┴────────┐                        │
        (pass)             (fail)                     │
           ▼                 ▼                        │
   [prediction_node]       [END]                      │
           │                                          │
           ├──────────────────────┬───────────────────┤
           │                      │                   │
           ▼                      ▼                   ▼
    [weather_node]           [soil_node]          [rag_node]
    (OpenWeatherMap/         (100 Synthetic       (FAISS Dense Search
      Open-Meteo)              Records, 5-Tier)     + Keyword Fallback)
           │                      │                   │
           │    [gradcam_node]    │                   │
           │   (features.16.0)    │                   │
           │          │           │                   │
           └──────────┼───────────┴───────────────────┘
                      ▼
          [context_aggregation_node]  <-- Synchronized Join Barrier
                      │
                      ▼
           [gemini_guidance_node]     <-- Single Cloud LLM Call Budget
                      │
                      ▼
                    [END]
```

### Architectural Safeguards in the Graph:
1. **Deterministic Join Barrier:** `context_aggregation_node` acts as a synchronization barrier, ensuring that parallel branches (`weather_node`, `soil_node`, `rag_node`, and optional `gradcam_node`) complete execution before formatting the context for Gemini.
2. **Quality Short-Circuit:** If an uploaded image fails Laplacian sharpness, exposure, or dimension thresholds, the workflow routes immediately to `END`, preventing unnecessary GPU/CPU inference and LLM calls.
3. **List Reducers for Concurrency:** `completed_stages` and `error_information` utilize `Annotated[list[str], operator.add]` reducers so parallel nodes return isolated additions without state overwriting collisions.
4. **Single Gemini Invocation:** Cloud LLM calls are strictly budget-capped at exactly one call per successful workflow execution.

---

## 3. Technology Stack

- **Orchestration & Workflow:** `langgraph>=0.2.0`, `langchain-core>=0.3.0`.
- **Deep Learning & Vision:** `torch>=2.2` (CPU-optimized), `torchvision>=0.17`, `Pillow>=10.0`, `numpy>=1.26.0`, `scipy>=1.11.0`.
- **Generative AI Layer:** Google Gemini API via the official `google-genai>=0.1.1` SDK (candidate chain: `gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash`).
- **RAG & Vector Retrieval:** `sentence-transformers>=3.0` (`all-MiniLM-L6-v2`), `faiss-cpu>=1.8` (Flat Inner Product cosine search).
- **Backend Framework:** FastAPI, Uvicorn (ASGI), Pydantic v2, Python-multipart, Requests, Python-dotenv.
- **Meteorological Services:** OpenWeatherMap API with automatic zero-key fallback to Open-Meteo API.
- **Voice Synthesis (TTS):** Microsoft Azure Neural voices (`mr-IN-AarohiNeural`, `en-IN-NeerjaExpressiveNeural`) via `edge-tts>=7.0.0` with `gTTS>=2.5` fallback and MD5-hashed disk caching.
- **Frontend Presentation:** Responsive HTML5, Vanilla JavaScript (ES6+), Glassmorphism CSS3 design system, Web Speech API for voice STT.
- **Testing:** `pytest>=8.0` (144 automated unit, integration, and reliability tests).

---

## 4. Dataset & Preprocessing

- **Dataset:** Curated Solanaceae subset of the **PlantVillage** dataset.
- **13 Supported Classes:**
  1. Bell Pepper — Bacterial spot (`Pepper__bell___Bacterial_spot`)
  2. Bell Pepper — Healthy (`Pepper__bell___healthy`)
  3. Potato — Early blight (`Potato___Early_blight`)
  4. Potato — Late blight (`Potato___Late_blight`)
  5. Tomato — Bacterial spot (`Tomato_Bacterial_spot`)
  6. Tomato — Early blight (`Tomato_Early_blight`)
  7. Tomato — Late blight (`Tomato_Late_blight`)
  8. Tomato — Leaf mold (`Tomato_Leaf_Mold`)
  9. Tomato — Septoria leaf spot (`Tomato_Septoria_leaf_spot`)
  10. Tomato — Spider mites (`Tomato_Spider_mites_Two_spotted_spider_mite`)
  11. Tomato — Target spot (`Tomato__Target_Spot`)
  12. Tomato — Tomato yellow leaf curl virus (`Tomato__Tomato_YellowLeaf__Curl_Virus`)
  13. Tomato — Healthy (`Tomato_healthy`)
- **Quarantining & Balance:** Severely underrepresented classes (`Potato___healthy` with only 152 images, `Tomato_mosaic_virus` with 373 images) were quarantined to preserve class balance.
- **Bounded Stream Reading:** FastAPI enforces a strict 10MB memory upload boundary (`image.read(MAX_FILE_SIZE_BYTES + 1)`), rejecting oversized uploads with HTTP 413 and empty files with HTTP 400.
- **Heuristic Quality Screening:**
  - Blur Detection: Laplacian kernel variance $\sigma^2$ (threshold $\ge 35.0$).
  - Exposure Check: Grayscale pixel luminance (acceptable bounds: $30.0 \le \mu \le 230.0$).
  - Dimension Boundaries: Minimum resolution $100 \times 100$, maximum $6000 \times 6000$.

---

## 5. Model Architecture & Explainability

- **Backbone:** ImageNet-pretrained **MobileNetV3-Large** (`models/mobilenetv3_best.pth`, 17.08 MB).
- **Custom Classifier Head:** `Linear(960, 512) -> Hardswish -> Dropout(0.2) -> Linear(512, 13)`.
- **Visual Explainability (Grad-CAM):** Forward and backward PyTorch hooks attach to layer `features.16.0` (final point-wise convolution before global average pooling). Normalizes gradient-weighted activation maps into a color heatmap overlaid onto the original leaf image and returns a base64-encoded PNG.

---

## 6. Soil Health Agent & 100-Record Synthetic Dataset

The **Soil Health Service** (`src/soil/soil_service.py`) grounds advisory in regional soil chemistry without demanding manual numeric test entries from the farmer:
- **Validated Dataset (`data/soil/synthetic_soil_records.json` & `.csv`):** Exactly 100 validated synthetic records across Maharashtra agro-climatic zones (Western Maharashtra, Scarcity Zone, Khandesh, Vidarbha, Marathwada, and state benchmarks).
- **Parameters Modeled:** `soil_ph`, `soil_moisture_percent`, `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`, `organic_carbon_percent`, `electrical_conductivity_ds_m`, `soil_type`, and `drainage_class`.
- **5-Tier Fallback Hierarchy:**
  1. *Tier 1:* Exact District + Exact Crop match (`match_method="exact"`).
  2. *Tier 2:* District + Similar Crop family (`match_method="district_crop_family"`).
  3. *Tier 3:* District-level average across all crops (`match_method="district_average"`).
  4. *Tier 4:* State-level agro-ecological zone average for the crop (`match_method="state_crop_average"`).
  5. *Tier 5:* Deterministic Maharashtra benchmark profile (`match_method="default_benchmark"`).
- **Academic Disclosure:** Every record, API response, and UI element carries an explicit notice: `"data_type": "synthetic"`, reminding farmers that data is for demonstration and not a substitute for laboratory core testing.

---

## 7. Knowledge Base & RAG Pipeline

- **Authoritative Source:** 13 curated markdown documents in `knowledge_base/` based on ICAR (IIHR Bengaluru, CPRI Shimla, IIVR Varanasi) and State Agricultural University publications.
- **Chunking & Index:** Parsed into 62 topic sections (`models/rag_chunks.json`) and embedded into a 384-dimensional dense FAISS index (`models/rag_index.faiss`) using `all-MiniLM-L6-v2`.
- **Keyword Fallback:** If FAISS is uninitialized or encounters an error, `AgriculturalRAG` automatically executes a lexical token-overlap search (`_keyword_search()`) directly over raw knowledge chunks.

---

## 8. Live Weather & Epidemiological Risk Rules

The weather module provides real-time meteorological conditions with automatic fallback:
- **Primary Provider:** OpenWeatherMap API.
- **Automatic Fallback:** Open-Meteo API (zero API key required).
- **Microclimate Disease Rules:**
  - *Late Blight:* Temp $15-22^\circ\text{C}$ + Humidity $\ge 80\%$.
  - *Early Blight & Bacterial Spot:* Temp $22-30^\circ\text{C}$ + Humidity $\ge 80\%$.
  - *Spider Mites:* Temp $> 28^\circ\text{C}$ + Humidity $< 45\%$.
  - *Powdery Mildew:* Temp $20-28^\circ\text{C}$ + Humidity $50-70\%$.

---

## 9. Tripartite Gemini Synthesis & Marathi Localization

The final advisory synthesizes the **Tripartite Context** (Disease Perception + Live Weather + Regional Soil):
- **Official SDK:** Uses `google-genai` with client `genai.Client(api_key=...)`.
- **Candidate Fallback Chain:** Automatically tries `gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash` upon encountering rate limits (HTTP 429).
- **Deterministic Offline RAG Fallback:** If all cloud API calls fail or offline mode is active, `_fallback_grounded_guidance()` synthesizes complete, structured guidance directly from retrieved RAG evidence chunks.
- **Centralized Localization (`src/utils/localization.py`):** Authoritative bilingual dictionaries and translation helpers ensuring consistency between backend responses and UI language toggles.

---

## 10. Project Directory Structure

```text
├── backend/
│   ├── main.py                     # FastAPI application, routes, lifespan & bounded uploads
│   └── __init__.py
├── data/
│   ├── samples/                    # 6 reference sample test images
│   └── soil/
│       ├── synthetic_soil_records.json # 100 synthetic Maharashtra soil records
│       └── synthetic_soil_records.csv
├── docs/
│   ├── BACKEND_ARCHITECTURE_GUIDE.md # Technical defense guide & subsystem deep-dive
│   ├── FLOWCHART.md                # System execution flowcharts & Mermaid diagrams
│   └── SYSTEM_ARCHITECTURE.md      # Comprehensive architecture specification
├── frontend/
│   ├── app.js                      # Frontend controller, Web Audio, DOM rendering
│   ├── index.html                  # Responsive semantic HTML5 single-page application
│   ├── styles.css                  # UI design tokens, glassmorphism, responsive rules
│   └── data/samples/               # Sample preview images for static hosting
├── knowledge_base/                 # 13 ICAR/TNAU markdown pathology documents
├── models/
│   ├── class_names.json            # 13 class label mappings
│   ├── mobilenetv3_best.pth        # Trained MobileNetV3-Large PyTorch weights (17.08 MB)
│   ├── rag_chunks.json             # 62 extracted knowledge base section chunks
│   └── rag_index.faiss             # Serialized FAISS vector index binary
├── outputs/
│   ├── audio_cache/                # Generated TTS audio cache (.gitkeep)
│   ├── classification_report.txt   # Empirical model validation report
│   ├── confusion_matrix.png        # 13x13 Seaborn confusion matrix plot
│   └── split_manifest.json         # Dataset 80/10/10 split log (1.85 MB)
├── src/
│   ├── data/                       # Offline dataset inspection and preparation scripts
│   ├── evaluation/                 # Offline model evaluation suite
│   ├── gemini/                     # Gemini guidance service & offline RAG fallback
│   ├── graph/                      # LangGraph StateGraph orchestration layer
│   │   ├── nodes.py                # 8 asynchronous node adapters
│   │   ├── state.py                # AgricultureAssistantState TypedDict & reducers
│   │   └── workflow.py             # Workflow construction, routing & compile
│   ├── prediction/                 # Vision inference (Top-3 predictions & Grad-CAM)
│   ├── preprocessing/              # Image quality gates & PyTorch transforms
│   ├── rag/                        # Agricultural RAG & FAISS vector store
│   ├── soil/                       # Soil Health Service & 5-tier fallback matcher
│   ├── training/                   # Model definition (build_mobilenetv3_large) & training
│   ├── utils/                      # Voice service (Edge TTS + gTTS) & localization
│   └── __init__.py
├── tests/                          # Automated Pytest regression test suite (144 tests)
├── .env.example                    # Template environment variables
├── .gitignore                      # Git ignore patterns
├── config.yaml                     # Central application configuration
├── requirements.txt                # Production runtime dependencies (with CPU PyTorch)
├── requirements-dev.txt            # Development, offline ML training & test dependencies
├── run.bat                         # Windows quickstart launcher
└── vercel.json                     # Vercel deployment configuration
```

---

## 11. Quick Start & Execution Guide

### Prerequisites
- Python 3.10 – 3.13 (Verified on Python 3.13.5).
- Git.

### 1. Installation
Clone the repository and set up a virtual environment:
```powershell
# Clone the repository
git clone https://github.com/karankarpe/AI-Agriculture-assistant.git
cd "AI Agriculture assistant final"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install production runtime dependencies
pip install -r requirements.txt

# (Optional) Install development and testing dependencies
pip install -r requirements-dev.txt
```

### 2. Configuration (`.env`)
Create a `.env` file from the provided example:
```env
GEMINI_API_KEY=your_gemini_api_key_here
WEATHER_API_KEY=your_openweathermap_api_key_here  # Optional: Open-Meteo works without a key
PORT=8000
HOST=0.0.0.0
```

### 3. Run the Backend & Frontend
Start the local FastAPI development server:
```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Web Application: **`http://localhost:8000/`**
- Interactive Swagger API Documentation: **`http://localhost:8000/docs`**

### 4. Running the Automated Test Suite
Execute the comprehensive 144-test regression suite:
```powershell
python -m pytest -o pythonpath=. -v
```
**Current Test Status:** `144 passed, 0 failed, 1 warning in ~70-90s`.

---

## 12. Deployment Architecture

| Tier | Platform | Build Command | Start Command | URL |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend** | **Vercel** | Static hosting | Automatic | [https://krishi-sahayyak.vercel.app](https://krishi-sahayyak.vercel.app) |
| **Backend** | **Render** | `pip install -r requirements.txt` | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` | [https://krishi-sahayyak.onrender.com](https://krishi-sahayyak.onrender.com) |

The frontend automatically detects whether it is running on `localhost` (routing requests to relative path `""`) or on Vercel (routing requests to the secured Render production backend `https://krishi-sahayyak.onrender.com`).

---

## 13. Documentation Suite

For detailed technical specifications, viva defense materials, and workflow diagrams:
- **[docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md):** Complete end-to-end system architecture, LangGraph state design, and multi-tier component breakdown.
- **[docs/BACKEND_ARCHITECTURE_GUIDE.md](docs/BACKEND_ARCHITECTURE_GUIDE.md):** Technical defense guide explaining lifespan startup, threadpool offloading, sub-2ms audio caching, and failure recovery.
- **[docs/FLOWCHART.md](docs/FLOWCHART.md):** Complete visual flowcharts covering `/analyze` and `/ask` LangGraph execution DAGs.

---

## 14. Academic Integrity & Ethics

1. **Synthetic Data Disclosure:** All soil records are synthetic baselines generated for academic illustration. Simulated demonstration values are never misrepresented as certified physical laboratory core samples.
2. **Safe Chemical Advisory:** Chemical recommendations adhere strictly to ICAR guidelines. The system explicitly instructs farmers to verify advice with local Krishi Vigyan Kendra (KVK) officers before field application.
3. **Controlled Image Limitations:** The perception model was trained on PlantVillage benchmark images. Real farm imagery with complex backgrounds, multiple overlapping leaves, or varied field lighting requires professional confirmation.
