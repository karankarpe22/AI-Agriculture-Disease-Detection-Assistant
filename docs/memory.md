# MEMORY.md — Persistent Project Memory & Knowledge Base

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Completed, Evaluated, and Deployed Production Baseline  
> **Target Audience**: Future AI Coding Agents, Maintainers, and Reviewers  
> **Last Verified**: September 2026  

---

## 1. Project Purpose & Scope

The **AI Agriculture Assistant** is an evidence-based clinical plant disease diagnostic and advisory platform developed as a 3rd-year engineering capstone project. Its objective is to provide high-accuracy visual disease detection for Indian smallholder crops (Tomato, Potato, Bell Pepper), contextualized with real-time weather risk analysis, authoritative agricultural package-of-practices from ICAR, and empathetic voice-enabled guidance in English and Marathi.

---

## 2. Verified Technical Facts (Ground Truth)

### 2.1 Model & Vision Weights
- **Base Architecture**: `torchvision.models.mobilenet_v3_large` initialized with ImageNet-1K pretrained weights (`MobileNet_V3_Large_Weights.DEFAULT`).
- **Input Dimension**: `(3, 224, 224)` tensor with ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Head Structure**: Replaced 1000-class head with:
  ```python
  Linear(in_features=960, out_features=1280) -> Hardswish() -> Dropout(p=0.2) -> Linear(in_features=1280, out_features=13)
  ```
- **Active Model Weights File**: `models/mobilenetv3_best.pth` (17,083,901 bytes / 16.29 MB).
- **Secondary Weights Files**:
  - `models/mobilenetv3_head_best.pth` (Head-only training checkpoint, 17,085,491 bytes).
  - `models/mobilenetv3_head_smoke_best.pth` (Initial smoke test checkpoint, 17,087,463 bytes).
- **Training Strategy**: Two-stage transfer learning (Stage 1: Frozen backbone, $lr=10^{-3}$; Stage 2: Top-3 InvertedResidual blocks unfrozen, $lr=10^{-4}$ with Cosine Annealing).

### 2.2 Dataset & Exact 13 Disease Classes
- **Dataset Source**: PlantVillage dataset curated subset.
- **Excluded Classes (Quarantined)**: `Potato___healthy` (only 152 images), `Tomato__Tomato_mosaic_virus` (373 images) were excluded due to extreme data imbalance.
- **Split Distribution**: Total: 20,113 images (Train: 14,080 [70%], Validation: 3,018 [15%], Test: 3,015 [15%]).
- **Exact Class Index Mapping (`models/class_names.json`)**:
  0. `Pepper__bell___Bacterial_spot` -> (Bell pepper, Bacterial spot)
  1. `Pepper__bell___healthy` -> (Bell pepper, Healthy)
  2. `Potato___Early_blight` -> (Potato, Early blight)
  3. `Potato___Late_blight` -> (Potato, Late blight)
  4. `Tomato_Bacterial_spot` -> (Tomato, Bacterial spot)
  5. `Tomato_Early_blight` -> (Tomato, Early blight)
  6. `Tomato_Late_blight` -> (Tomato, Late blight)
  7. `Tomato_Leaf_Mold` -> (Tomato, Leaf mold)
  8. `Tomato_Septoria_leaf_spot` -> (Tomato, Septoria leaf spot)
  9. `Tomato_Spider_mites_Two_spotted_spider_mite` -> (Tomato, Spider mites [Two-spotted spider mite])
  10. `Tomato__Target_Spot` -> (Tomato, Target spot)
  11. `Tomato__Tomato_YellowLeaf__Curl_Virus` -> (Tomato, Tomato yellow leaf curl virus)
  12. `Tomato_healthy` -> (Tomato, Healthy)

### 2.3 Measured Experimental Results (`outputs/evaluation.json`)
- **Test Accuracy**: **81.89%** ($2,469 / 3,015$ correct on independent test split).
- **Macro Precision**: **0.8699**
- **Macro Recall**: **0.8170**
- **Macro F1-Score**: **0.8164**
- **Weighted F1-Score**: **0.8139**
- **Average Model Confidence**: **91.00%**
- **Average CPU Inference Latency**: **21.58 ms / image**

### 2.4 Agricultural Knowledge Base & RAG
- **Markdown Source Files**: 13 files in `knowledge_base/` (`bell_pepper_bacterial_spot.md`, `bell_pepper_healthy.md`, `potato_early_blight.md`, `potato_late_blight.md`, `tomato_bacterial_spot.md`, `tomato_early_blight.md`, `tomato_healthy.md`, `tomato_late_blight.md`, `tomato_leaf_mold.md`, `tomato_septoria_leaf_spot.md`, `tomato_spider_mites.md`, `tomato_target_spot.md`, `tomato_yellow_leaf_curl_virus.md`).
- **Pre-computed Chunk File**: `models/rag_chunks.json` (62 semantic sections, 50,061 bytes).
- **Retrieval Engine**: Instant structured metadata and keyword relevance scorer in `src/rag/vector_store.py`. Operates with zero memory overhead at inference time, preventing 512 MB RAM container kills.
- **FAISS Cache**: `models/rag_index.faiss` (95,277 bytes) exists for offline index generation.

### 2.5 Weather Integration
- **Implementation**: `src/weather/weather_service.py`.
- **Primary Provider**: OpenWeatherMap API (`https://api.openweathermap.org/data/2.5/weather`).
- **Automatic Fallback**: Open-Meteo REST API (`https://api.open-meteo.com/v1/forecast`), requiring zero API keys.
- **Microclimate Analysis**: Flags high fungal sporulation risk when humidity $> 80\%$ and temperature is between $20^\circ	ext{C}$ and $30^\circ	ext{C}$.

### 2.6 Generative AI Guidance
- **Implementation**: `src/gemini/guidance_service.py`.
- **SDK**: Official `google-genai` Python library (`from google import genai`).
- **Active Model**: `gemini-2.5-flash`.
- **Safety Boundary**: Receives MobileNetV3 prediction and ICAR evidence as strict ground truth. Never diagnoses the leaf independently.
- **Offline Fallback Engine**: `_fallback_grounded_guidance()` delivers structured ICAR guidance even if Gemini API quota is exhausted.

### 2.7 Voice & Accessibility
- **STT**: Browser Web Speech API (`webkitSpeechRecognition`).
- **TTS**: `src/utils/voice_service.py` using `gTTS` (`lang='mr'` for Marathi, `lang='en'` with `tld='co.in'` for Indian English).
- **Audio Cache**: Generated MP3 files stored in `outputs/audio_cache/`.

### 2.8 Backend & Web Server
- **Framework**: FastAPI with Pydantic v2 schemas in `backend/main.py`.
- **Local Startup**: `run.bat` or `run.ps1` executes `uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload`.
- **Cloud Deployment**: Live at `https://krishi-sahayyak.onrender.com` (Uvicorn running on Python 3.12+).

---

## 3. Important File Directory Map

```text
D:\GenAI MiniProject\AI Agriculture assistant final├── backend/
│   └── main.py                <- FastAPI router, endpoints, static file mounting
├── frontend/
│   ├── index.html             <- Glassmorphic user interface structure
│   ├── styles.css             <- Design system, HSL tokens, animations, responsive layout
│   └── app.js                 <- UI state, dynamic API_BASE, I18N dictionary, voice STT/TTS
├── knowledge_base/            <- 13 peer-reviewed ICAR markdown guidance documents
├── models/
│   ├── class_names.json       <- Ordered 13 class labels
│   ├── mobilenetv3_best.pth   <- Production PyTorch model weights (16.29 MB)
│   ├── rag_chunks.json        <- 62 pre-parsed semantic sections with metadata
│   └── rag_index.faiss        <- Pre-built FAISS vector index
├── outputs/
│   ├── evaluation.json        <- Independent test split metrics (81.89% acc)
│   ├── confusion_matrix.png   <- 13x13 test set confusion matrix
│   └── audio_cache/           <- Cached gTTS MP3 files
├── src/
│   ├── data/                  <- Dataset inspection, splitting, and stats tools
│   ├── evaluation/            <- Evaluation scripts for test splits
│   ├── gemini/                <- Google GenAI guidance layer and fallback
│   ├── prediction/            <- DiseasePredictor and Grad-CAM generator
│   ├── preprocessing/         <- Image quality validator and PyTorch transforms
│   ├── rag/                   <- AgriculturalRAG retrieval service
│   ├── training/              <- Model architecture and training loops
│   ├── utils/                 <- VoiceService (gTTS)
│   └── weather/               <- Dual-provider weather service
├── tests/                     <- 24 automated unit and integration tests (100% pass)
├── .env                       <- Local environment variables (DO NOT COMMIT)
├── .env.example               <- Template for API keys
├── config.yaml                <- Core project configuration and class mappings
├── README.md                  <- Project overview, architecture, and academic results
├── requirements.txt           <- Python dependencies
├── run.bat                    <- One-click Windows launch script
└── run.ps1                    <- PowerShell launch script
```

---

## 4. Known Limitations & Edge Cases

1. **Lab vs. Field Domain Gap**: PlantVillage leaves were imaged under controlled laboratory illumination with uniform paper backgrounds. Real-world leaves photographed with complex field backgrounds, harsh sunlight, or extreme shadows may yield lower model confidence.
2. **Co-occurring Pathologies**: The current model classifies the single dominant pathology (multi-class single-label classification). It does not perform multi-label segmentation if a leaf suffers from both Early Blight and Spider Mite damage simultaneously.
3. **Render Free Tier Spin-Down**: On Render's free tier, the web service spins down after 15 minutes of inactivity. The first subsequent request can take 30–50 seconds to cold-start.

---

## 5. Invariants That Must NOT Be Changed Without Explicit Justification

1. **Do not reorder or alter the 13 classes in `models/class_names.json`**. Doing so corrupts the model's output index-to-label mapping.
2. **Do not replace MobileNetV3-Large with an un-pretrained or custom CNN architecture**.
3. **Do not allow Gemini to classify the image**. The separation of perception (PyTorch) and generation (Gemini) is the project's foundational architectural rule.
4. **Do not load heavy transformer models (`sentence-transformers`) at inference time**. The zero-RAM lexical scorer must remain active to prevent OOM container terminations on 512 MB hosting.
5. **Do not commit `.env` or hardcode API keys**.
