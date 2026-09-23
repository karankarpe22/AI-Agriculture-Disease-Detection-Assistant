# TRD.md — Technical Requirements Document

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Target Version**: 1.0.0 Production  
> **Document Status**: Complete & Verified Against Source Implementation  

---

## 1. System Architecture Overview

The system is architected as an asynchronous, decoupled, multi-tier Python web application:
- **Client Tier**: Responsive single-page web application (HTML5, Vanilla CSS3, ES6+ JavaScript).
- **Application Tier**: FastAPI (Python 3.12+ ASGI) orchestrating perceptual vision inference, contextual environmental lookup, agricultural knowledge retrieval, and LLM guidance synthesis.
- **Inference Engine**: PyTorch MobileNetV3-Large running on CPU with sub-30ms inference latency.
- **Knowledge Layer**: 62-chunk indexed ICAR document corpus with zero-RAM structured relevance scoring.
- **Cloud Hosting**: Render Free Web Service (`https://krishi-sahayyak.onrender.com`) and local workstation deployment via `run.bat`.

---

## 2. Technology Stack & Exact Dependencies

All dependency requirements are pinned in `requirements.txt`:

| Category | Library / Dependency | Version Specifier | Purpose in System |
| :--- | :--- | :--- | :--- |
| **Language Runtime** | Python | `>=3.12, <3.14` | Core runtime (tested on 3.12.14 and 3.13.5) |
| **Machine Learning** | `torch` | `>=2.2` (CPU wheel) | Deep learning framework, tensor compute, autograd |
| | `torchvision` | `>=0.17` | Vision models, ImageNet pretrained weights, transforms |
| | `scikit-learn` | `>=1.3` | Evaluation metrics (precision, recall, F1, confusion matrix) |
| | `Pillow` | `>=10.0` | Image loading, resizing, EXIF handling, quality checks |
| | `PyYAML` | `>=6.0` | Parsing `config.yaml` |
| | `matplotlib` | `>=3.8` | Plotting training histories and confusion matrix heatmaps |
| | `seaborn` | `>=0.13` | Statistical data visualization |
| **RAG & Vector Retrieval** | `sentence-transformers` | `>=3.0` | Offline vector embedding generation |
| | `faiss-cpu` | `>=1.8` | Flat Inner Product vector index for semantic search |
| **Backend & Web API** | `fastapi` | `>=0.110` | High-performance asynchronous REST API framework |
| | `uvicorn[standard]` | `>=0.28` | Production ASGI web server |
| | `pydantic` | `>=2.6` | Request/response data validation and settings schemas |
| | `python-multipart` | `>=0.0.9` | Multipart form parsing for file uploads |
| | `requests` | `>=2.31` | Synchronous HTTP client for weather APIs |
| | `python-dotenv` | `>=1.0` | Local `.env` configuration management |
| **Generative AI** | `google-genai` | `>=0.1.1` | Official Google GenAI Python SDK for Gemini 2.5 Flash |
| **Speech & Audio** | `gTTS` | `>=2.5` | Google Text-to-Speech synthesis for Marathi and English |
| **Testing** | `pytest` | `>=8.0` | Automated test runner |

---

## 3. Machine Learning Specifications

### 3.1 Model Architecture Details
- **Architecture**: `MobileNetV3-Large`
- **Pretrained Source**: `torchvision.models.MobileNet_V3_Large_Weights.DEFAULT` (ImageNet-1K pretrained)
- **Input Dimensions**: Tensor shape `(B, 3, 224, 224)` normalized by:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Feature Extractor Output**: 960 channels after Global Average Pooling.
- **Custom Classifier Head**:
  ```text
  Sequential(
    (0): Linear(in_features=960, out_features=1280, bias=True)
    (1): Hardswish()
    (2): Dropout(p=0.2, inplace=True)
    (3): Linear(in_features=1280, out_features=13, bias=True)
  )
  ```
- **Total Parameters**: $\sim 4.2 	ext{ Million}$
- **Active Weights Checkpoint**: `models/mobilenetv3_best.pth` ($17,083,901 	ext{ bytes} pprox 16.29 	ext{ MB}$)

### 3.2 Dataset Structure & Class Schema
- **Dataset**: Curated subset of the **PlantVillage** dataset.
- **Total Dataset Size**: 20,113 images across 13 classes.
- **Splits**: 70% Train (14,080), 15% Validation (3,018), 15% Test (3,015) partitioned with `random_seed: 42`.
- **Target Classes**:
  1. `Pepper__bell___Bacterial_spot`
  2. `Pepper__bell___healthy`
  3. `Potato___Early_blight`
  4. `Potato___Late_blight`
  5. `Tomato_Bacterial_spot`
  6. `Tomato_Early_blight`
  7. `Tomato_Late_blight`
  8. `Tomato_Leaf_Mold`
  9. `Tomato_Septoria_leaf_spot`
  10. `Tomato_Spider_mites_Two_spotted_spider_mite`
  11. `Tomato__Target_Spot`
  12. `Tomato__Tomato_YellowLeaf__Curl_Virus`
  13. `Tomato_healthy`

### 3.3 Training Methodology
- **Loss Function**: `CrossEntropyLoss()`
- **Stage 1 (Head Training)**: Backbone frozen (`requires_grad = False`), AdamW optimizer with $lr = 1 	imes 10^{-3}$, weight decay $1 	imes 10^{-4}$, `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)`.
- **Stage 2 (Fine-Tuning)**: Top-3 InvertedResidual blocks unfrozen (`features[13:]`), AdamW with $lr = 1 	imes 10^{-4}$, `CosineAnnealingLR(T_max=10, eta_min=1e-6)`.
- **Batch Size**: 64 images per batch.

### 3.4 Measured Empirical Evaluation Metrics
Evaluated on the independent test partition (3,015 unseen images):
- **Overall Accuracy**: **81.89%** ($2,469 / 3,015$)
- **Macro Precision**: **0.8699**
- **Macro Recall**: **0.8170**
- **Macro F1-Score**: **0.8164**
- **Weighted F1-Score**: **0.8139**
- **Average Model Confidence**: **91.00%**
- **CPU Inference Latency**: **21.58 ms / sample**

---

## 4. RAG, Weather & External API Specifications

### 4.1 Knowledge Base & Retrieval
- **Knowledge Base Location**: `knowledge_base/` containing 13 verified Markdown guides.
- **Pre-parsed Chunks**: `models/rag_chunks.json` (62 semantic sections, 50 KB).
- **Retrieval Algorithm**: Multi-factor metadata scoring:
  - Exact/Partial Crop Match: $+6.0$
  - Disease Token Overlap: $+5.0$ per token
  - Actionable Heading Bias (Management/Treatment): $+3.0$
  - Question Keyword Overlap: $+1.5$ per keyword
- **RAM Overhead**: 0 MB extra memory consumed during request handling.

### 4.2 Weather Service
- **Primary Provider**: OpenWeatherMap API v2.5 (`https://api.openweathermap.org/data/2.5/weather`) via `WEATHER_API_KEY`.
- **Fallback Provider**: Open-Meteo REST API (`https://api.open-meteo.com/v1/forecast`), zero authentication required.
- **Heuristic Pathogen Pressure Formula**:
  - High Risk: Relative Humidity $> 80\%$ and Temperature between $18^\circ	ext{C}$ and $30^\circ	ext{C}$ (accelerates fungal germination).
  - Moderate Risk: Humidity $60\% - 80\%$.
  - Low Risk: Humidity $< 60\%$.

### 4.3 Google Gemini Integration
- **SDK**: `google-genai` (v0.1.1+)
- **Model**: `gemini-2.5-flash`
- **Authentication**: `GEMINI_API_KEY` from environment variables.
- **Input Parameters**: Temperature 0.2 (low variance, strict grounding), JSON schema enforcement.
- **Fallback Strategy**: Deterministic ICAR Markdown template synthesizer (`_fallback_grounded_guidance`).

---

## 5. Security & Environmental Controls

- **Secret Isolation**: All credentials loaded strictly from process environment (`os.getenv("GEMINI_API_KEY")`, `os.getenv("WEATHER_API_KEY")`).
- **Git Controls**: `.env` is ignored by `.gitignore`.
- **Upload Restrictions**: Max file upload size: $10 	ext{ MB}$. Enforced MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/bmp`.
- **CORS Configuration**: Configured via `fastapi.middleware.cors.CORSMiddleware`.

---

## 6. Hosting & Deployment Requirements

### 6.1 Local Workstation (Windows)
- **Launch Command**: `run.bat` or `run.ps1`
- **Host / Port**: `127.0.0.1:8000`
- **Virtual Environment**: `.venv-windows` (Python 3.12+)

### 6.2 Cloud Host (Render.com)
- **Service Type**: Web Service (Free Tier)
- **Runtime**: Python 3 (Ubuntu 22.04 LTS container)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Resource Constraints**: 512 MB RAM, 0.1 CPU core.
- **Live URL**: `https://krishi-sahayyak.onrender.com`

---

## 7. Performance & Latency Benchmarks

| Operation | Target Budget | Measured Performance | Verification Method |
| :--- | :--- | :--- | :--- |
| **Image Validation** | $< 15 	ext{ ms}$ | $4.2 	ext{ ms}$ | Laplacian variance compute benchmark |
| **MobileNetV3 Inference** | $< 35 	ext{ ms}$ | $21.58 	ext{ ms}$ | PyTorch CPU test split benchmark |
| **RAG Retrieval** | $< 10 	ext{ ms}$ | $0.8 	ext{ ms}$ | Metadata scoring across 62 chunks |
| **Weather Fetch** | $< 400 	ext{ ms}$ | $180 	ext{ ms}$ | HTTP request to OpenWeatherMap / Open-Meteo |
| **Gemini Synthesis** | $< 2500 	ext{ ms}$ | $1200 - 1800 	ext{ ms}$ | Live Gemini 2.5 Flash streaming API |
| **Total `/analyze` Pipeline** | $< 3000 	ext{ ms}$ | $1600 - 2200 	ext{ ms}$ | End-to-end integration test timing |
