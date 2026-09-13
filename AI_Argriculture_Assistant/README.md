# AI Agriculture Assistant (कृषी सहाय्यक)

An academically grounded, evidence-based agricultural decision support system designed for 3rd-year engineering research. It integrates **MobileNetV3-Large** for transfer-learning-based crop disease classification, a **FAISS-grounded RAG (Retrieval-Augmented Generation)** knowledge layer from ICAR publications, **real-time weather risk analysis**, and **Google Gemini** generative advisory in **English and Marathi**.

---

## 1. Project Overview & Problem Statement
Smallholder farmers in India frequently face crop losses due to delayed or inaccurate diagnosis of foliar diseases. While modern computer vision models can classify leaf pathology, standalone classifiers lack contextual explanation, cannot account for microclimatic weather risks (such as humidity accelerating fungal sporulation), and cannot answer farmer queries in regional languages. Conversely, generic large language models (LLMs) hallucinate pesticide recommendations when asked to diagnose images directly.

The **AI Agriculture Assistant** resolves this by decoupling perception from generative guidance:
1. An ImageNet-pretrained **MobileNetV3-Large** model identifies the plant disease and provides a calibrated confidence score.
2. A lightweight **RAG pipeline** extracts authoritative treatment and management practices from curated ICAR (Indian Council of Agricultural Research) and agricultural university guides.
3. A **Weather Service** incorporates live ambient temperature, relative humidity, and precipitation.
4. **Google Gemini** synthesizes these multimodal signals into safe, practical farmer guidance in English or Marathi without hallucinating chemical dosages.

---

## 2. System Architecture Pipeline

```text
               +-----------------------------+
               |       Crop Leaf Image       |
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |     Image Quality Check     |
               | (Blur, Exposure, Resolution)|
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               |    MobileNetV3-Large CNN    |
               |  (ImageNet Pretrained Head) |
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               | Disease + Confidence Score  |
               +-----------------------------+
                              |
        +---------------------+---------------------+
        |                                           |
        v                                           v
+-------------------+                       +---------------+
|  Agricultural RAG |                       |  Live Weather |
| (FAISS + MiniLM)  |                       |  Microclimate |
+-------------------+                       +---------------+
        |                                           |
        +---------------------+---------------------+
                              |
                              v
               +-----------------------------+
               |  Farmer Question + Context  |
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               | Google Gemini Generative AI |
               | (Strict Evidence Grounding) |
               +-----------------------------+
                              |
                              v
               +-----------------------------+
               | English / Marathi Guidance  |
               | (Audio Voice TTS & Text)    |
               +-----------------------------+
```

---

## 3. Technology Stack

- **Machine Learning & Deep Learning**: PyTorch 2.14, Torchvision 0.29, MobileNetV3-Large (`MobileNet_V3_Large_Weights.DEFAULT`), scikit-learn.
- **RAG & Vector Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), `faiss-cpu` (Flat Inner Product cosine search).
- **Generative AI Layer**: Google Gemini API via official `google-genai` Python SDK (`gemini-2.5-flash`).
- **Weather Services**: OpenWeatherMap API with automatic live fallback to Open-Meteo API (zero key required).
- **Backend API**: FastAPI, Uvicorn, Pydantic v2, Python-multipart.
- **Frontend UI**: Responsive HTML5, Vanilla CSS3 (glassmorphism design system, Outfit & Noto Sans Devanagari fonts), JavaScript (ES6+), Web Speech API for voice STT.
- **Voice Synthesis (TTS)**: `gTTS` for Marathi (`mr`) and English (`en`) audio streaming.

---

## 4. Dataset & Preprocessing

- **Dataset**: Curated subset of the publicly accessible **PlantVillage** dataset.
- **Crops & Classes (13 Classes)**:
  1. Bell Pepper - Bacterial spot (`Pepper__bell___Bacterial_spot`)
  2. Bell Pepper - Healthy (`Pepper__bell___healthy`)
  3. Potato - Early blight (`Potato___Early_blight`)
  4. Potato - Late blight (`Potato___Late_blight`)
  5. Tomato - Bacterial spot (`Tomato_Bacterial_spot`)
  6. Tomato - Early blight (`Tomato_Early_blight`)
  7. Tomato - Late blight (`Tomato_Late_blight`)
  8. Tomato - Leaf mold (`Tomato_Leaf_Mold`)
  9. Tomato - Septoria leaf spot (`Tomato_Septoria_leaf_spot`)
  10. Tomato - Spider mites (`Tomato_Spider_mites_Two_spotted_spider_mite`)
  11. Tomato - Target spot (`Tomato__Target_Spot`)
  12. Tomato - Tomato yellow leaf curl virus (`Tomato__Tomato_YellowLeaf__Curl_Virus`)
  13. Tomato - Healthy (`Tomato_healthy`)
- **Quarantining**: Excluded severely underrepresented classes (`Potato___healthy` with only 152 images, `Tomato_mosaic_virus` with 373 images) to preserve baseline statistical balance.
- **Data Split**: Class-stratified 70% Train (14,080 images), 15% Validation (3,018 images), 15% Test (3,015 images). Exact byte-duplicates were detected and grouped to avoid train-test data leakage.
- **Transforms**:
  - Training: Resize to $224 \times 224$, RandomHorizontalFlip ($p=0.5$), RandomRotation ($\pm 15^\circ$), ColorJitter (brightness/contrast $0.15$), ImageNet normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
  - Validation/Testing: Deterministic Resize ($224 \times 224$) and ImageNet normalization.

---

## 5. Transfer Learning & Fine-Tuning Methodology

A scientifically defensible two-stage training strategy was applied:
- **Stage 1 (Head Training)**: The MobileNetV3-Large backbone was frozen. The ImageNet 1000-class classifier was replaced with a new linear head for 13 classes. Trained with AdamW ($lr=1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$) and ReduceLROnPlateau scheduler.
- **Stage 2 (Fine-Tuning)**: The top 3 InvertedResidual blocks of the backbone were unfrozen. Trained with a reduced learning rate ($lr=1 \times 10^{-4}$) and Cosine Annealing decay to adjust high-level feature representations without destroying pretrained ImageNet weights.
- **Checkpointing**: Checkpoint saved based on validation accuracy (`models/mobilenetv3_best.pth`).

---

## 6. Actual Experimental Results (Measured, Never Fabricated)

Evaluated on the independent test split (**3,015 unseen images** across 13 classes):

| Metric | Measured Value |
| :--- | :--- |
| **Test Accuracy** | **81.89%** |
| **Macro Precision** | **0.8699** |
| **Macro Recall** | **0.8170** |
| **Macro F1-Score** | **0.8164** |
| **Weighted F1-Score** | **0.8139** |
| **Average Confidence** | **91.00%** |
| **CPU Inference Latency** | **21.58 ms / image** |
| **Model Checkpoint Size** | **16.29 MB** |

### Per-Class Highlights
- **Bell Pepper Healthy**: Precision 0.85, Recall 1.00, F1 0.92
- **Potato Early Blight**: Precision 0.95, Recall 0.94, F1 0.95
- **Potato Late Blight**: Precision 0.95, Recall 0.93, F1 0.94
- **Tomato TYLCV**: Precision 0.99, Recall 0.91, F1 0.95
- **Tomato Healthy**: Precision 0.95, Recall 0.94, F1 0.94

Artifacts generated:
- `outputs/evaluation.json`: Complete quantitative metrics.
- `outputs/confusion_matrix.png`: 13x13 Seaborn confusion matrix.
- `outputs/classification_report.txt`: Full precision/recall/F1 breakdown.
- `outputs/finetuning_history.png`: Loss and accuracy curves.

---

## 7. Image Quality Check & Uncertainty Estimation

Before classification, every image passes an automated quality gate:
- **Resolution**: Rejects images smaller than $100 \times 100$ px.
- **Blur Detection**: Calculates Laplacian variance $\sigma^2$; flags images with $\sigma^2 < 35.0$ as blurry.
- **Exposure**: Evaluates grayscale mean; flags overexposed ($>230$) or underexposed ($<30$) images.
- **Uncertainty Warning**: If softmax confidence is $< 60\%$, the system displays an explicit cautionary alert advising on-site consultation with a Krishi Vigyan Kendra (KVK) officer.
- **Explainability**: Optional **Grad-CAM** generates an attention heatmap overlay on the leaf.

---

## 8. Agricultural Knowledge Base & RAG

- **Knowledge Base**: 13 curated markdown documents in `knowledge_base/` sourced from ICAR (IIHR Bengaluru, CPRI Shimla, IIVR Varanasi) and TNAU.
- **Sections**: Symptoms, Favorable Microclimates, Cultural Management, Biological Control, Recommended Fungicides/Acaricides, and Pre-Harvest Intervals.
- **Index**: Built using `sentence-transformers/all-MiniLM-L6-v2` and cached into a FAISS index (`models/rag_index.faiss`, 62 indexed sections).
- **Retrieval**: Embeds composite query `"{crop} {disease} {farmer_question}"` to fetch top-$k$ relevant chunks with citations.

---

## 9. Live Weather & Microclimate Risk Analysis

The weather module queries real-time meteorological conditions by city or coordinates:
- Temperature, relative humidity, precipitation, wind speed, and conditions.
- Primary provider: OpenWeatherMap (via `WEATHER_API_KEY`).
- Live zero-key fallback: **Open-Meteo API** (works out of the box).
- **Epidemiological Risk Rules**:
  - Humidity $\ge 80\%$ + Temp $15-22^\circ\text{C}$: High risk for Late Blight (*Phytophthora infestans*).
  - Humidity $\ge 80\%$ + Temp $22-30^\circ\text{C}$: High risk for Early Blight and Bacterial Spot.
  - Humidity $< 45\%$ + Temp $> 28^\circ\text{C}$: High risk for Spider Mite proliferation.

---

## 10. Gemini Integration & Marathi Support

- Uses official Google GenAI SDK (`google-genai`).
- System prompt enforces:
  1. Do NOT claim image diagnosis; take classifier output as given.
  2. Ground recommendations strictly in the retrieved RAG evidence.
  3. Integrate current weather factors.
  4. Never invent pesticide dosages.
  5. Support fluent **English** and **Marathi (मराठी)**.
- Offline Fallback: If `GEMINI_API_KEY` is not provided, an integrated evidence synthesis engine compiles the exact RAG citations and weather warnings directly into structured guidance.

---

## 11. Voice Accessibility (Speech-to-Text & Text-to-Speech)

- **Speech-to-Text**: Native browser Web Speech API (`webkitSpeechRecognition`) supporting Marathi (`mr-IN`) and English (`en-IN`).
- **Text-to-Speech**: Modular `gTTS` audio generation via `POST /tts` endpoint, streaming MP3 audio directly to the frontend player.

---

## 12. Quick Start & Execution Guide

### Prerequisites
- Python 3.10 - 3.13 (Tested on Python 3.12/3.13 on Windows).

### 1. Installation
```powershell
# Activate virtual environment
.\.venv-windows\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create or edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
WEATHER_API_KEY=your_openweathermap_key_here  # Optional: Open-Meteo works automatically without key
```

### 3. How to Train
```powershell
# Stage 1: Train classification head
python -m src.training.train_head --epochs 3 --max-train-per-class 150 --max-validation-per-class 50

# Stage 2: Fine-tune upper backbone
python -m src.training.train_finetune --epochs 2 --max-train-per-class 150 --max-validation-per-class 50
```

### 4. How to Evaluate
```powershell
# Evaluate model on 3,015 test images
python -m src.evaluation.evaluate --test-dir data/test
```

### 5. How to Run Backend & Web Interface
```powershell
# Start FastAPI server on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser and navigate to:
**`http://localhost:8000/`**

Interactive API documentation:
**`http://localhost:8000/docs`**

### 6. Running the Test Suite
```powershell
pytest -v
```
*All 24 unit and integration tests will execute.*

---

## 13. Academic Limitations & Future Work

1. **Controlled vs. Field Images**: PlantVillage photos were captured under controlled laboratory settings. Real farm imagery with soil clutter, mixed lighting, and multiple leaves requires external field validation before commercial deployment.
2. **Confidence Calibration**: Raw softmax confidence does not equal true posterior probability. While temperature scaling and low-confidence thresholds help, expert verification remains necessary.
3. **Pesticide Safety**: Chemical recommendations must always be verified with local agricultural officers (Krishi Vigyan Kendra) to adhere to state-specific chemical bans and current crop stage restrictions.
4. **Future Extensions**: Implement edge quantization (TensorFlow Lite / ONNX Mobile), offline on-device speech recognition via Whisper-tiny, and expanding coverage to cotton, sugarcane, and soybean.
