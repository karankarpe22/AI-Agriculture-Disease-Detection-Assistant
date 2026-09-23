# AGENTS.md — AI Coding Agent Operating Guide

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Existing & Fully Implemented Production System  
> **Target Platform**: Local Windows (`run.bat` on `http://localhost:8000`) & Cloud Web Service (`https://krishi-sahayyak.onrender.com`)  
> **Core Architecture**: PyTorch MobileNetV3-Large + ICAR-Grounded RAG + Live Weather Risk + Google Gemini Generative Guidance + gTTS Voice + Bilingual Glassmorphic UI  

---

## 1. Project Identity & Mission

The **AI Agriculture Assistant** is an academically grounded foliar disease detection and farmer guidance system designed for Indian smallholder agriculture. It bridges the gap between raw computer vision classification and contextual, actionable farmer advisory.

The central engineering philosophy of this project is **Decoupled Perception and Generation**:
1. **Perception**: A deterministic, fine-tuned deep learning model (**MobileNetV3-Large**) classifies plant leaf pathology into 13 calibrated classes.
2. **Contextual Grounding**: An evidence-based **RAG engine** retrieves authoritative package-of-practices from Indian Council of Agricultural Research (**ICAR**) and state agricultural university publications, combined with live microclimate weather data.
3. **Synthesis**: **Google Gemini** synthesizes the diagnostic, environmental, and retrieved evidence into accessible, farmer-friendly recommendations in English or Marathi.

---

## 2. Current Implementation Status

This is **NOT** a greenfield or hypothetical project. The entire pipeline is fully developed, tested, and operational:
- **Trained Model**: Checkpoint weights exist at `models/mobilenetv3_best.pth` (16.29 MB) achieving **81.89% test accuracy** and **0.8164 Macro F1** across 3,015 test images.
- **Agricultural RAG**: 62 curated sections indexed from 13 comprehensive ICAR markdown documents in `knowledge_base/`, with pre-computed chunk index at `models/rag_chunks.json`.
- **Backend API**: Production FastAPI service in `backend/main.py` exposing 10 endpoints (`/`, `/health`, `/predict`, `/analyze`, `/weather`, `/ask`, `/tts`, `/styles.css`, `/app.js`, `/favicon.ico`).
- **Frontend UI**: Glassmorphic, bilingual (English / मराठी) interface in `frontend/` with drag-and-drop leaf upload, voice recognition (STT), voice synthesis (TTS), and Grad-CAM visual attention heatmaps.
- **Automated Test Suite**: 24 unit, integration, and API tests in `tests/` with a **100% pass rate**.

---

## 3. Core Development Principles for AI Agents

Every AI coding agent operating in this repository **must** strictly adhere to the following workflow:

```text
+----------+      +------+      +--------+      +------+      +--------+      +----------+
| INSPECT  | ---> | PLAN | ---> | MODIFY | ---> | TEST | ---> | VERIFY | ---> | DOCUMENT |
+----------+      +------+      +--------+      +------+      +--------+      +----------+
```

### The "Never Rebuild" Golden Rule
- **NEVER** delete, reset, recreate, or replace working components.
- **NEVER** restart from scratch or propose migrating to an entirely different architecture (e.g., swapping FastAPI for Django, replacing PyTorch with TensorFlow, or replacing MobileNetV3 with an untrained custom CNN).
- **NEVER** introduce parallel duplicate implementations (e.g., creating `backend_v2/` or `new_predict.py`).
- Continue strictly from the **CURRENT IMPLEMENTATION**.

---

## 4. Step-by-Step Agent Modification Protocol

### Step 1: Inspect Before Modifying
Before touching any source file:
1. Read the target module and all related callers.
2. Check `config.yaml` for relevant constants, class lists, and hyperparameters.
3. Check `tests/` to understand the existing invariants and contracts.
4. Verify what dependencies are present in `requirements.txt`.

### Step 2: Plan Incrementally
1. Formulate the smallest possible patch that satisfies the user's requirement.
2. Ensure backward compatibility: existing API response schemas, class indices, and UI bindings must remain unbroken.
3. Identify potential side effects on memory (e.g., keeping cloud host RAM strictly below 512 MB).

### Step 3: Modify with Surgical Precision
1. Modify only the targeted lines or methods. Preserve existing comments, docstrings, and error handling.
2. Never hardcode API keys, personal credentials, or machine-specific absolute file paths.
3. Keep all file paths relative or resolved via `Path(__file__).resolve().parent`.

### Step 4: Test & Verify
1. Run the targeted unit test:
   ```bash
   python -m pytest tests/test_<component>.py -v
   ```
2. Run the full regression test suite:
   ```bash
   python -m pytest tests/ -v
   ```
3. Verify server startup:
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
4. Confirm `GET /health` returns `{"status": "healthy"}`.

### Step 5: Document
1. Record any modified behavior in `memory.md` and `walkthrough.md`.
2. If configuration changed, update `config.yaml` and `.env.example`.

---

## 5. Domain-Specific Component Guidelines

### A. Machine Learning Layer (`src/prediction/`, `src/training/`, `models/`)
- **Class Schema Invariance**: The 13 disease classes are fixed by the trained weights in `models/mobilenetv3_best.pth`. Never reorder or alter `models/class_names.json` without retraining the model.
- **Input Dimension**: All vision transforms must produce tensors of shape `[3, 224, 224]` normalized with standard ImageNet statistics ($\\mu = [0.485, 0.456, 0.406]$, $\\sigma = [0.229, 0.224, 0.225]$).
- **Inference Mode**: Always wrap inference in `with torch.no_grad():` and ensure `model.eval()` is set.
- **Grad-CAM**: Grad-CAM targets the last convolutional layer of MobileNetV3 (`model.features[-1]`). Do not break layer hooking.

### B. Agricultural RAG Layer (`src/rag/`, `knowledge_base/`)
- **Zero-RAM Footprint**: The production RAG retrieval uses an instant, deterministic metadata-and-keyword relevance scorer across `models/rag_chunks.json`. **Do not** re-introduce heavy transformer models (such as loading `sentence-transformers` in memory at inference time) on memory-constrained hosting tiers (Render Free 512 MB RAM).
- **Factual Grounding**: All retrieved advice must originate from curated ICAR/TNAU markdown documents in `knowledge_base/`. Never fabricate synthetic agricultural recommendations.

### C. Generative Advisory Layer (`src/gemini/`)
- **Strict Role Separation**: Gemini must **never** diagnose the leaf or override the MobileNetV3 classification. It receives `crop`, `disease`, `confidence`, `weather`, and `retrieved_evidence` as established facts and generates an accessible advisory.
- **SDK Usage**: Use the official Google GenAI Python SDK (`from google import genai`). The active model is `gemini-2.5-flash`.
- **Fallback Engine**: Always maintain the rule-based fallback generator (`_fallback_grounded_guidance`) so the application delivers ICAR recommendations even during Gemini API outages or quota limits.

### D. Backend API Layer (`backend/main.py`)
- **Schema Contracts**: Endpoints `/predict`, `/analyze`, `/weather`, `/ask`, and `/tts` are consumed by `frontend/app.js`. Do not rename JSON response keys without updating the frontend.
- **Resilience**: Wrap external network calls (weather APIs, Gemini API) in defensive `try/except` blocks so that downstream network failures degrade gracefully rather than throwing 500 Internal Server Errors.

### E. Frontend Layer (`frontend/`)
- **Vanilla Modern Web**: Maintain standard HTML5, CSS3, and ES6+ JavaScript. Do not introduce heavy frontend frameworks (React, Vue, Angular) or complex build steps (Webpack, Vite) unless explicitly requested by the user.
- **Bilingual Localization**: All UI strings must be registered in the `I18N` translation dictionary inside `frontend/app.js` for both English and Marathi (`मराठी`).
- **Dynamic API Base**: Maintain dynamic `API_BASE` resolution to ensure compatibility across local development (`http://localhost:8000`), local file opening, and cloud HTTPS deployment.

---

## 6. Prohibited Actions & Strict Constraints

| Action | Policy | Rationale |
| :--- | :---: | :--- |
| **Deleting or overwriting existing model weights** | 🚫 **STRICTLY FORBIDDEN** | `mobilenetv3_best.pth` represents completed multi-epoch training and validation. |
| **Fabricating evaluation metrics** | 🚫 **STRICTLY FORBIDDEN** | All metrics (81.89% accuracy, 0.8164 F1) are verified experimental test results. |
| **Committing API keys or `.env`** | 🚫 **STRICTLY FORBIDDEN** | Secrets must remain isolated in environment variables. |
| **Allowing Gemini to classify the image** | 🚫 **STRICTLY FORBIDDEN** | Violates the core decoupled perception architecture and causes LLM hallucination. |
| **Modifying dataset splits without documentation** | 🚫 **STRICTLY FORBIDDEN** | Train (70%), validation (15%), and test (15%) splits are fixed in `split_manifest.json`. |
| **Bypassing automated test verification** | 🚫 **STRICTLY FORBIDDEN** | All 24 tests must pass after any functional modification. |
