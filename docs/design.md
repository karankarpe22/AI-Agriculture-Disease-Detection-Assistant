# DESIGN.md — Detailed System & Interaction Design

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Implemented & Operational  
> **Primary Interface**: Glassmorphic Multilingual Web UI  
> **Backend Architecture**: RESTful Asynchronous Microservice  

---

## 1. End-to-End Operational Workflow

The system executes a deterministic 9-stage sequence upon receiving an image from the farmer:

```text
       [ Farmer uploads leaf photo + location + query ]
                             |
                             v
                 [ 1. Input Validation ]
            (MIME type, size <= 10MB check)
                             |
                             v
               [ 2. Image Quality Analysis ]
           (Resolution >= 128px, Brightness, Blur)
                             |
                             v
               [ 3. MobileNetV3 Inference ]
           (ImageNet normalized forward pass)
                             |
                             v
              [ 4. Confidence & Risk Scoring ]
       (Calculate softmax probabilities & threshold check)
                             |
                             v
             [ 5. Agricultural Knowledge RAG ]
        (Retrieve top-3 ICAR chunks for crop & disease)
                             |
                             v
             [ 6. Environmental Weather Risk ]
      (Fetch temperature, humidity, precipitation & risk)
                             |
                             v
            [ 7. Google Gemini Advisory Synthesis ]
       (Evidence-grounded prompt compilation in EN or MR)
                             |
                             v
             [ 8. Grad-CAM Attention Heatmap ]
      (Optional backward gradient pass on final Conv2D)
                             |
                             v
        [ 9. Response Delivery & Voice Generation ]
      (Glassmorphic UI display + gTTS streaming audio)
```

---

## 2. Detailed Flow Specifications

### 2.1 Upload Flow (`frontend/app.js` & `backend/main.py`)
- **Trigger**: Farmer drags-and-drops an image onto `#dropzone` or clicks the file selector.
- **Client-Side Checks**: Checks file existence, displays immediate thumbnail preview in `#preview-img`, and hides the dropzone prompt.
- **Payload Composition**: Appends `image` binary, `location` string (default "Pune"), `question` string (optional), `language` ("english" or "marathi"), and `include_gradcam` boolean flag to a `FormData` object.
- **Server-Side Validation**: `validate_image_file()` in `backend/main.py` enforces:
  - Max file size: $10 	ext{ MB}$ ($10 	imes 1024 	imes 1024$ bytes).
  - Permitted extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.
  - Permitted MIME types: `image/jpeg`, `image/png`, `image/webp`, `image/bmp`.

### 2.2 Image Quality Assurance Flow (`src/preprocessing/quality_check.py`)
- **Resolution Assessment**: Verifies $	ext{width} \ge 128$ and $	ext{height} \ge 128$.
- **Photometric Assessment**: Converts image to grayscale; calculates mean pixel luminance $\mu_L$:
  - $\mu_L < 40.0$: Flagged as `Too Dark` (underexposed).
  - $\mu_L > 230.0$: Flagged as `Too Bright` (overexposed/washed out).
- **Sharpness Assessment**: Computes the Laplacian operator variance $\sigma^2_{	ext{Laplace}}$:
  - $\sigma^2_{	ext{Laplace}} < 50.0$: Flagged as `Excessively Blurry`.
- **Early Rejection**: If `is_valid == False`, the pipeline aborts deep learning inference and immediately returns an informative user warning guiding the farmer to retake the photo in daylight.

### 2.3 Disease Prediction Flow (`src/prediction/predict.py`)
- **Image Preprocessing**: Image is resized to $224 	imes 224$ and normalized using PyTorch torchvision transforms ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Tensor Generation**: Unsqueezed to batch shape `[1, 3, 224, 224]` on target device (`cpu` or `cuda`).
- **Forward Pass**: Forward pass through `build_mobilenetv3_large()` with weights loaded from `models/mobilenetv3_best.pth`.
- **Softmax Probability**: Computes `probs = torch.softmax(logits, dim=1)`.
- **Top-K Extraction**: Extracts top-3 candidates (`class_index`, `raw_class`, `crop`, `disease`, `confidence`).

### 2.4 Confidence & Uncertainty Flow
- **Threshold**: Set to $	au = 0.60$ ($60\%$).
- **High Confidence ($\ge 60\%$)**: Marked as confident prediction; display badge in green.
- **Low Confidence ($< 60\%$)**: Marked with `is_low_confidence = True`.
- **Warning Banner Generation**: Generates an advisory banner: *"Prediction confidence is low. Symptoms may be in early stages or obscured. Please verify with a local agricultural extension officer."*
- **Alternative Candidates Display**: Exposes collapsible `<details>` panel in the UI listing the second and third most likely disease candidates with their respective percentage probabilities.

### 2.5 Agricultural Knowledge RAG Flow (`src/rag/vector_store.py`)
- **Corpus Loading**: Reads `models/rag_chunks.json` (62 semantic chunks).
- **Composite Scoring**: Each chunk is evaluated dynamically:
  $$	ext{Score} = S_{	ext{crop}} + S_{	ext{disease}} + S_{	ext{query}} + S_{	ext{heading}}$$
  - $S_{	ext{crop}} = +6.0$ if predicted crop matches chunk metadata.
  - $S_{	ext{disease}} = +5.0$ per disease keyword matched.
  - $S_{	ext{query}} = +1.5$ per farmer question token match.
  - $S_{	ext{heading}} = +3.0$ for Treatment/Management, $+1.5$ for Symptoms, $+1.0$ for Environmental Conditions.
- **Deduplicated Top-3**: Chunks are ranked descending and deduplicated across document sections to guarantee diverse, high-value evidence.
- **Citation Extraction**: Compiles authoritative citations (e.g., *"ICAR - Indian Institute of Horticultural Research (IIHR)"*).

### 2.6 Environmental Weather Risk Flow (`src/weather/weather_service.py`)
- **City Geocoding**: Resolves location string (e.g., "Nashik") to geographic coordinates ($	ext{lat}, 	ext{lon}$).
- **API Call**: Queries OpenWeatherMap Current Weather endpoint. If rate-limited or unconfigured, seamlessly queries Open-Meteo REST API.
- **Metrics Collected**: Temperature ($^\circ	ext{C}$), Relative Humidity ($\%$), Precipitation ($	ext{mm}$), Wind Speed ($	ext{km/h}$).
- **Microclimate Pathogen Pressure Analysis**:
  - High humidity ($> 80\%$) + warm temperatures ($20 - 30^\circ	ext{C}$) flags elevated fungal sporulation risk (e.g., Late Blight, Leaf Mold).
  - High humidity + wind flags rapid bacterial dissemination risk (e.g., Bacterial Spot).

### 2.7 Generative Guidance Flow (`src/gemini/guidance_service.py`)
- **Model Target**: Google Gemini via `google-genai` SDK (`gemini-2.5-flash`).
- **Strict Grounding Rules**: The system instruction forbids Gemini from altering the disease diagnosis or recommending pesticides outside the retrieved ICAR evidence.
- **Structured JSON Schema**:
  ```json
  {
    "explanation": "Botanical description of the identified symptoms...",
    "weather_interpretation": "Impact of local temperature and humidity on disease spread...",
    "management_guidance": "Recommended ICAR cultural and biological practices...",
    "prevention": "Preventative sanitation and crop rotation steps...",
    "precautions": "Safety measures for handling sprays and irrigation...",
    "uncertainty_warning": "Warning text if confidence is below 60%...",
    "expert_advisory": "Contact details for local extension centers...",
    "language": "english",
    "powered_by": "Google Gemini & ICAR Knowledge Base"
  }
  ```
- **Fallback Rule Engine**: If Gemini API returns an error or quota exhaustion, `_fallback_grounded_guidance()` deterministically converts the ICAR markdown chunks into the exact same JSON format, guaranteeing uninterrupted service.

### 2.8 Grad-CAM Explainability Flow (`src/prediction/predict.py`)
- **Target Layer**: Last depthwise-separable convolutional layer of MobileNetV3 (`model.features[-1]`).
- **Gradient Computation**: Backward pass computed with respect to the winning class score.
- **Feature Map Weighting**: Global average pooling of gradients weights the activation feature maps.
- **Overlay Rendering**: Normalizes heatmap, applies OpenCV-compatible Jet color mapping, and blends with original image at $40\%$ alpha transparency. Returned to client as base64 JPEG data URL.

### 2.9 Multilingual & Voice Flow (`frontend/app.js` & `src/utils/voice_service.py`)
- **Language Switch**: Click on English or मराठी toggle buttons updates client state and swaps all static text tokens via `I18N`.
- **Speech-to-Text (STT)**: Web Speech API listens to farmer's spoken query via microphone button, populating `#farmer-question`.
- **Text-to-Speech (TTS)**: Clicking "Listen to Advisory" triggers `POST /tts` with guidance text.
- **Server Audio Synthesis**: `gTTS` generates streaming MP3 audio:
  - English voice: `tld='co.in'`, `lang='en'`
  - Marathi voice: `lang='mr'`
- **Audio Caching**: Audio clips are hashed by MD5 and saved to `outputs/audio_cache/` for instantaneous replay without redundant network calls.

---

## 3. Frontend Screen Architecture

```text
+----------------------------------------------------------------------------------------------------+
|  [Logo] AI Agriculture Assistant | कृषी सहाय्यक                    [Status Badge] [EN | मराठी]     |
|         Evidence-Grounded Crop Disease Detection & Advisory                                        |
+----------------------------------------------------------------------------------------------------+
|                                                  |                                                 |
|  LEFT PANEL: Input & Perception                  |  RIGHT PANEL: Diagnosis & Guidance              |
|                                                  |                                                 |
|  +--------------------------------------------+  |  +-------------------------------------------+  |
|  | [DROPZONE / LEAF IMAGE PREVIEW]            |  |  | [Disease Heading: e.g. Tomato Early Blight] |  |
|  | Drag & drop or click to upload leaf photo  |  |  | [Confidence Pill: 94.2% (High Confidence)]  |  |
|  +--------------------------------------------+  |  +-------------------------------------------+  |
|                                                  |                                                 |
|  Location: [ Nashik                         ]    |  [Weather Widget: 26°C | 78% Humidity | Rain ]  |
|  Quick Pills: [Pune] [Nashik] [Nagpur] [Kolhapur]|  Microclimate: High fungal spread pressure      |
|                                                  |                                                 |
|  Farmer Question:                                |  [Explanation Card]                             |
|  [ e.g. What organic spray should I use?    ]    |  Concentric brown rings detected on foliage...  |
|  [Microphone Voice Button]                       |                                                 |
|                                                  |  [Management & Treatment Card]                  |
|  [x] Generate Grad-CAM Visual Attention Heatmap  |  1. Spray Mancozeb 75 WP @ 2.5 g/L (ICAR-IIVR)  |
|                                                  |  2. Remove and safely burn infected debris      |
|  [ BUTTON: Analyze Crop Health               ]   |                                                 |
|                                                  |  [Prevention & Precautions Card]                |
|                                                  |  Avoid overhead irrigation; practice rotation   |
|                                                  |                                                 |
|                                                  |  [Listen to Audio Advisory (MP3 Player)]        |
|                                                  |  [Grad-CAM Attention Heatmap Image]             |
|                                                  |  [Authoritative Citations & Source Links]       |
+----------------------------------------------------------------------------------------------------+
```

---

## 4. Backend Endpoint Interaction Matrix

| Endpoint | Method | Input Parameters | Output Schema | Primary Consumer |
| :--- | :---: | :--- | :--- | :--- |
| `/` | `GET` | None | `text/html` | Browser client entry point |
| `/health` | `GET` | None | `HealthResponse` JSON | System health badge |
| `/weather` | `GET` | `city`, `lat`, `lon` | Weather JSON + Risk analysis | Standalone weather lookup |
| `/predict` | `POST` | `image: UploadFile` | Classification JSON + Top-3 | Computer vision verification |
| `/ask` | `POST` | `AskRequest` JSON | Gemini Guidance JSON | Direct advisory querying |
| `/analyze` | `POST` | `multipart/form-data` | Full composite analysis JSON | Main "Analyze Crop Health" button |
| `/tts` | `POST` | `text`, `language` | `audio/mpeg` binary stream | "Listen to Advisory" button |
