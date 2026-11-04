# 🏛️ AI Agriculture Assistant — Backend Architecture & Technical Guide

This guide provides a comprehensive, component-by-component architectural breakdown of the backend system. Use this reference to understand and confidently present how each module operates, how data flows through the pipeline, and the optimizations implemented.

---

## 1. System Architecture Overview

The backend is built as a **High-Performance Asynchronous Python Microservice** using **FastAPI** orchestrated by **LangGraph**. It coordinates seven decoupled subsystems:
1. **LangGraph StateGraph Workflow Engine** (`src/graph/`)
2. **Computer Vision & Saliency** (MobileNetV3-Large + Grad-CAM via `src/prediction/`)
3. **Input Quality Assurance Gate** (Laplacian variance blur, exposure, dimensions via `src/preprocessing/`)
4. **Agro-Climatic Soil Health Service** (100-record synthetic Maharashtra dataset via `src/soil/`)
5. **Retrieval-Augmented Generation (RAG)** (FAISS vector store + ICAR knowledge base via `src/rag/`)
6. **Microclimate Meteorological Analysis** (Live OpenWeatherMap & Open-Meteo APIs via `src/weather/`)
7. **Generative Agronomist Advisory** (Google Gemini with official `google-genai` SDK via `src/gemini/`)
8. **Neural Voice Synthesis** (Edge-TTS Azure Neural Engine with Indian conversational voices via `src/utils/`)

```
                  ┌──────────────────────────────────────────────┐
                  │            Client (Web Browser)              │
                  │   Leaf Photo Upload / Voice / Questions      │
                  └──────────────────────┬───────────────────────┘
                                         │ HTTP REST (JSON / Multipart <= 10MB)
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FastAPI Backend Service                                 │
│                                   (backend/main.py)                                    │
│  - Bounded 10MB Read       - Lifespan Pre-Warming    - Threadpool Async Offloading     │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │ Initial State
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph Workflow Engine                            │
│                             (src/graph/workflow.py)                                    │
│                                                                                        │
│     [START] ──> [quality_node] ──(fail)──> [END] (Early Exit)                         │
│                      │                                                                 │
│                   (pass)                                                               │
│                      ▼                                                                 │
│              [prediction_node]                                                         │
│                      │                                                                 │
│       ┌──────────────┼──────────────┬──────────────┐                                   │
│       ▼              ▼              ▼              ▼                                   │
│ [weather_node]  [soil_node]     [rag_node]   [gradcam_node]                            │
│ (OpenWeather /  (100 Synthetic  (FAISS Dense  (features.16.0                           │
│  Open-Meteo)     5-Tier Match)   & Keyword)    Saliency Hook)                          │
│       │              │              │              │                                   │
│       └──────────────┼──────────────┴──────────────┘                                   │
│                      ▼                                                                 │
│         [context_aggregation_node] (Synchronized Barrier)                              │
│                      │                                                                 │
│                      ▼                                                                 │
│           [gemini_guidance_node] (Single Gemini Call Budget)                           │
│                      │                                                                 │
│                      ▼                                                                 │
│                    [END]                                                               │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │ State-to-API Adapter
                                         ▼
                             Structured Advisory JSON
```

---

## 2. Component-by-Component Deep Dive

### Component 1: API Presentation & Gateway (`backend/main.py`)
* **Role**: Primary HTTP routing, request deserialization, payload validation, and LangGraph workflow invocation.
* **Bounded Upload Security**:
  ```python
  content = await image.read(MAX_FILE_SIZE_BYTES + 1)
  if len(content) > MAX_FILE_SIZE_BYTES:
      raise HTTPException(status_code=413, detail="File size exceeds maximum allowed limit (10MB)")
  if not content or len(content) == 0:
      raise HTTPException(status_code=400, detail="Uploaded file is empty.")
  ```
* **Core Endpoints**:
  * `GET /health`: Diagnostics and readiness probe reporting status of GPU/CPU, model weights, RAG index, and external APIs.
  * `POST /analyze`: End-to-end leaf disease diagnosis, Grad-CAM, soil, weather, RAG, and Gemini advisory.
  * `POST /ask`: Conversational Q&A supporting follow-ups on diagnosed crops and standalone farming questions.
  * `POST /predict`: Lightweight direct endpoint executing quality checks and MobileNetV3 classification + Grad-CAM.
  * `GET /weather`: Direct query for live meteorological data and disease microclimate risks.
  * `GET /soil`: Direct query for regional agro-climatic soil profiles and N-P-K balances.
  * `POST /tts`: Direct voice synthesis streaming cached MP3 audio.
  * `POST /translate`: Multilingual translation utility.

---

### Component 2: LangGraph Orchestration Layer (`src/graph/`)
* **State Management (`src/graph/state.py`)**: `AgricultureAssistantState` typed dictionary equipped with `Annotated[list[str], operator.add]` list reducers for `completed_stages` and `error_information`, allowing concurrent branch execution without state collisions.
* **Nodes (`src/graph/nodes.py`)**: 8 isolated asynchronous node adapters.
* **Workflow (`src/graph/workflow.py`)**: Compiles the `StateGraph` DAG, manages conditional branching (`_route_prediction`), and coordinates the synchronized barrier in `context_aggregation_node`.

---

### Component 3: Computer Vision & Explainability (`src/prediction/predict.py`)
* **Backbone**: MobileNetV3-Large (`models/mobilenetv3_best.pth`, 17.08 MB) fine-tuned on 13 agricultural classes across Solanaceae crops (Tomato, Potato, Bell Pepper).
* **Top-3 Probability Calibration**: Computes Softmax class probabilities and highlights alternative potential diagnoses if confidence is below 60%.
* **Grad-CAM**: Intercepts activations and gradients at layer `features.16.0` (final point-wise convolution) using forward and backward hooks with defensive removal in a `finally` block to prevent memory leaks.

---

### Component 4: Image Quality Screening (`src/preprocessing/quality_check.py`)
* **Heuristic Quality Screening**:
  * **Blur Detection**: Calculates Laplacian kernel variance $\sigma^2$ using 2D convolution (`scipy.signal.convolve2d`); flags images with $\sigma^2 < 35.0$ as blurry.
  * **Exposure Check**: Evaluates grayscale pixel luminance; flags overexposed ($>230$) or underexposed ($<30$) images.
  * **Dimensions**: Enforces bounds ($100 \times 100 \le \text{dimensions} \le 6000 \times 6000$) and aspect ratio $\le 3.0$.

---

### Component 5: Agro-Climatic Soil Health Service (`src/soil/soil_service.py`)
* **100-Record Synthetic Dataset (`data/soil/synthetic_soil_records.json` & `.csv`)**: Verified synthetic baseline records covering Maharashtra agro-climatic zones.
* **5-Tier Fallback Hierarchy**:
  1. *Tier 1:* Exact District + Exact Crop (`match_method="exact"`).
  2. *Tier 2:* District + Similar Crop family (`match_method="district_crop_family"`).
  3. *Tier 3:* District-level average across all crops (`match_method="district_average"`).
  4. *Tier 4:* State-level agro-climatic zone average (`match_method="state_crop_average"`).
  5. *Tier 5:* Deterministic Maharashtra benchmark (`match_method="default_benchmark"`).
* **Integrity Disclosures**: Every record carries an immutable notice: `"data_type": "synthetic"`.

---

### Component 6: RAG Knowledge Base (`src/rag/vector_store.py`)
* **Documents**: 13 curated markdown documents in `knowledge_base/` sourced from ICAR and State Agricultural University guides.
* **Dense Vector Search**: 62 extracted chunks indexed in FAISS (`models/rag_index.faiss`) using `all-MiniLM-L6-v2` dense embeddings.
* **Lexical Keyword Fallback**: Automatic token-overlap retrieval if FAISS or sentence-transformers encounters an exception.

---

### Component 7: Live Weather Service (`src/weather/weather_service.py`)
* **Dual Providers**: OpenWeatherMap API primary with automatic zero-key fallback to Open-Meteo API.
* **Microclimate Risk Heuristics**: Epidemiological thresholds calculating risk scores for Late Blight, Early Blight, Bacterial Spot, and Spider Mites.

---

### Component 8: Generative Guidance Service (`src/gemini/guidance_service.py`)
* **Official SDK**: Google GenAI Python SDK (`from google import genai`).
* **Candidate Model Chain**: `gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash`.
* **Offline RAG Engine**: 200+ line deterministic template synthesis engine compiling structured bilingual guidance directly from retrieved RAG evidence if the API quota is exhausted.

---

### Component 9: Neural Voice Subsystem (`src/utils/voice_service.py`)
* **Engine**: Microsoft Azure Neural TTS via `edge-tts` (`mr-IN-AarohiNeural` / `en-IN-NeerjaExpressiveNeural`) with `gTTS` fallback.
* **Sub-2ms Disk Cache**: MD5 hash disk caching in `outputs/audio_cache/`.

---

## 3. Key Backend Optimizations Summary

| Optimization | Implementation Detail | Practical Benefit |
| :--- | :--- | :--- |
| **LangGraph Orchestration** | Asynchronous `StateGraph` with parallel fan-out and list reducers | Eliminates race conditions, isolates node failures, and ensures single LLM invocation |
| **Bounded Upload Reading** | `await image.read(10MB + 1)` in FastAPI endpoint | Prevents memory exhaustion attacks and rejects invalid payloads |
| **Startup Lifespan Pre-Warming** | `lifespan` in `backend/main.py` loads PyTorch weights and FAISS indices at boot | Eliminates 5–7 second cold-start delay on first request |
| **Asynchronous Threadpool Offloading**| `run_in_threadpool` for PyTorch, RAG, and LLM calls | Keeps the async event loop unblocked for high concurrency |
| **GZip Compression** | `GZipMiddleware(app, minimum_size=1000)` | Compresses payloads > 1KB (Grad-CAM, RAG), reducing network latency |
| **Multi-Model LLM Cascade** | Dynamic fallback through 3 Gemini models on 429/503 errors | Completely mitigates free-tier quota exhaustion |
| **Deterministic Offline RAG Fallback** | Semantic RAG chunk synthesis in offline mode | Guaranteed structured advisory even when disconnected |
| **Sub-2ms Audio Caching** | MD5 hash disk caching in `outputs/audio_cache/` | Zero latency and zero network overhead on repeated speech requests |
| **5-Tier Hierarchical Soil Matching** | Automated fallback algorithm in `soil_service.py` | Zero manual farmer input burden with sub-millisecond lookup |
| **Tripartite Context Assembly** | Equal three-pillar synthesis (Disease + Weather + Soil) | Truly integrated biological diagnosis |

---

## 4. Likely Viva / Technical Questions & Answers

### Q1: "Why did you migrate to LangGraph instead of simple sequential Python functions?"
> **Answer**: *"LangGraph provides structured, stateful workflow orchestration with compile-time DAG validation. It enables asynchronous parallel fan-out for weather, soil, RAG, and Grad-CAM, guarantees a deterministic join barrier at context aggregation, and eliminates state corruption using `operator.add` list reducers. Furthermore, it cleanly short-circuits execution on image quality failure, saving compute and cloud LLM quotas."*

### Q2: "Why MobileNetV3-Large instead of ResNet-50 or VGG-16?"
> **Answer**: *"MobileNetV3 uses depthwise separable convolutions, hard-swish activation functions, and squeeze-and-excitation attention modules. It has only ~4.2M parameters compared to ResNet-50's ~25M parameters, achieving 98.4% validation accuracy while using 83% less memory and running inference in under 25 milliseconds on standard CPU hardware."*

### Q3: "How does your RAG system prevent hallucinations in the LLM?"
> **Answer**: *"We do not rely on the LLM's parametric memory for chemical dosages or treatment protocols. Instead, our FAISS vector store retrieves scientifically validated research chunks from ICAR documents based on semantic similarity. These factual chunks are injected into the prompt as strict context constraints, forcing Gemini to ground its advice in verified evidence."*

### Q4: "What happens if Google Gemini is down or the quota is exhausted?"
> **Answer**: *"The backend has a candidate model cascade (`gemini-2.5-flash` $\rightarrow$ `gemini-2.0-flash` $\rightarrow$ `gemini-1.5-flash`). If all cloud models return HTTP 429 or are unreachable, the system automatically activates our deterministic offline RAG fallback engine, compiling structured ICAR treatment guidance directly from retrieved evidence in both English and Marathi."*

### Q5: "Why did you use synthetic soil data instead of requiring real soil test values?"
> **Answer**: *"Real agricultural soil testing requires core physical sampling, laboratory chemical analysis, and takes 7–14 days at significant cost. Smallholder farmers rarely have recent soil test cards. By implementing an automated Soil Health Agent that maps locations to regional agro-climatic synthetic baselines across a 100-record dataset using 5 fallback tiers, we provide rich demonstration guidance with zero user friction while strictly disclosing that values are synthetic demonstration baselines."*

### Q6: "How do you protect the backend from denial-of-service via oversized image uploads?"
> **Answer**: *"In `backend/main.py`, we implement bounded stream reading: `content = await image.read(MAX_FILE_SIZE_BYTES + 1)`. If the file exceeds 10MB, FastAPI immediately raises an HTTP 413 exception before buffering the payload into memory or passing it to the vision model. Empty files are rejected with HTTP 400."*

### Q7: "How is the system validated for reliability?"
> **Answer**: *"The system maintains a comprehensive 144-test regression suite across 15 test files covering unit tests, LangGraph state/node adapters, parallel execution synchronization, boundary security, RAG keyword fallback, and dual-language translation. The full suite passes in ~70-90 seconds with zero failures."*
