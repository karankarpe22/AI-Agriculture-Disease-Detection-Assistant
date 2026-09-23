# DECISIONS.md — Key Architectural & Engineering Decisions

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Recorded & Active  
> **Standard Format**: Decision | Reason | Alternatives Considered | Current Status  

---

### Decision 1: MobileNetV3-Large as the Computer Vision Backbone

- **Decision**: Select **MobileNetV3-Large** (`torchvision.models.mobilenet_v3_large`) as the core feature extractor and classifier.
- **Reason**: MobileNetV3-Large provides an optimal balance between parameter efficiency (4.2M parameters, 16.29 MB checkpoint size), inference speed (~21.58 ms on commodity CPU), and representation capacity for fine-grained botanical features (such as subtle leaf spots, chlorosis, and concentric target lesions).
- **Alternatives Considered**:
  - *ResNet-50*: High accuracy, but bulky (~98 MB checkpoint, ~90 ms CPU latency), making it less suitable for low-resource server tiers.
  - *EfficientNet-B0*: Comparable accuracy, but slower CPU inference time due to depthwise separable convolutions without hardware-specific kernel tuning.
  - *Vision Transformers (ViT-Base)*: High accuracy on massive datasets, but prone to severe overfitting on small agricultural datasets and requires GPU acceleration.
- **Current Status**: **Active & Implemented** in `src/training/model.py` and `src/prediction/predict.py`.

---

### Decision 2: Using ImageNet Pretrained Weights (`MobileNet_V3_Large_Weights.DEFAULT`)

- **Decision**: Initialize the backbone using official PyTorch ImageNet-1K pretrained weights.
- **Reason**: ImageNet pretraining equips the lower and middle convolutional layers with generalized edge, texture, contour, and gradient detectors. This accelerates convergence on leaf images by orders of magnitude compared to random initialization.
- **Alternatives Considered**:
  - *Random Initialization (Training from Scratch)*: Required 5-10x more training epochs, severely overfitted on uniform lab backgrounds, and achieved poor generalization.
- **Current Status**: **Active & Implemented** in `src/training/model.py`.

---

### Decision 3: Two-Stage Transfer Learning with Backbone Fine-Tuning

- **Decision**: Execute a two-stage training methodology: (1) Freeze backbone and train the custom 13-class head with AdamW ($lr=10^{-3}$), then (2) Unfreeze top-3 InvertedResidual blocks with reduced learning rate ($lr=10^{-4}$) and Cosine Annealing scheduler.
- **Reason**: Training the head first avoids propagating large gradient errors into pretrained weights during early epochs. Selectively unfreezing top layers allows the model to adapt high-level visual representations specifically to foliar pathology without catastrophic forgetting.
- **Alternatives Considered**:
  - *Head-Only Linear Probing*: Underfitted; generic ImageNet features alone could not sufficiently distinguish subtle cross-pathogen textures (e.g., Early Blight vs. Late Blight).
  - *Full End-to-End Fine-Tuning*: Overfitted rapidly and destroyed low-level edge filters.
- **Current Status**: **Active & Implemented** in `src/training/train_head.py` and `src/training/train_finetune.py`.

---

### Decision 4: PlantVillage Dataset with a Curated 13-Class Focused Subset

- **Decision**: Restrict the training and validation scope to 13 high-impact classes across 3 vital Indian staple and cash crops: Tomato (8 diseases + healthy), Potato (2 diseases), and Bell Pepper (1 disease + healthy).
- **Reason**: PlantVillage is an open-access, academically recognized benchmark. However, the complete 38-class dataset contains extreme class imbalances. Two classes (`Potato___healthy` with only 152 images, and `Tomato_mosaic_virus` with 373 images) were deliberately quarantined to maintain statistical balance across the test set (3,015 test samples).
- **Alternatives Considered**:
  - *Full 38-Class PlantVillage*: Extreme imbalance led to near-zero recall on minority classes and dilute diagnostic focus.
  - *Custom Field Dataset*: Unlabeled and insufficient volume for defensible academic rigor at current project stage.
- **Current Status**: **Active & Implemented** in `config.yaml` and `src/data/prepare_dataset.py`.

---

### Decision 5: Why Not Train a Custom CNN Architecture from Scratch

- **Decision**: Reject custom, un-pretrained CNN architectures (e.g., a simple 4-layer Conv2D model).
- **Reason**: Training from scratch on ~14,000 images causes convolutional filters to memorize specific lab lighting, background tiles, and leaf orientations rather than pathological lesions, resulting in poor out-of-distribution generalization.
- **Alternatives Considered**:
  - *Custom 4-Layer ConvNet*: Overfitted within 8 epochs; failed completely on blurry or rotated leaves.
- **Current Status**: **Active Decision Preserved**.

---

### Decision 6: Lightweight Lexical/Metadata RAG over In-Memory Vector Transformers

- **Decision**: Use a zero-RAM structured relevance scoring algorithm across curated ICAR chunks, rather than keeping a 350 MB `SentenceTransformer` neural model permanently resident in memory during web inference.
- **Reason**: The knowledge base consists of 62 curated, high-density ICAR sections. For a domain-specific corpus of 62 chunks with known crop and disease taxonomy, structured lexical matching yields 100% precision in 0.5 ms with 0 MB extra RAM. In contrast, neural transformers caused out-of-memory (OOM) crashes on 512 MB cloud hosting tiers.
- **Alternatives Considered**:
  - *Always-On SentenceTransformer MiniLM in RAM*: Consumed 350 MB RAM; caused container kill on free-tier cloud servers.
  - *External Vector DB (Pinecone/Milvus)*: Unnecessary network latency, extra API costs, and operational overhead for 62 static documents.
- **Current Status**: **Active & Implemented** in `src/rag/vector_store.py`.

---

### Decision 7: Google Gemini as the Generative Layer (Not the Vision Classifier)

- **Decision**: Use **Google Gemini (`gemini-2.5-flash`)** strictly for synthesizing evidence, interpreting weather risks, and composing bilingual advice, while explicitly forbidding it from diagnosing the leaf image directly.
- **Reason**: LLMs hallucinate plant pathology when classifying images directly. By passing the ground-truth prediction from MobileNetV3 and verified ICAR chunks to Gemini as non-negotiable prompt context, hallucination is eliminated while natural, empathetic advisory is preserved.
- **Alternatives Considered**:
  - *End-to-End Gemini Vision (Gemini diagnosing the leaf)*: High rate of false positives on subtle fungal spots and potential for dangerous chemical dosage hallucinations.
  - *Hardcoded Template Responses*: Rigid, robotic, cannot incorporate weather nuance, and cannot answer dynamic farmer questions.
- **Current Status**: **Active & Implemented** in `src/gemini/guidance_service.py`.

---

### Decision 8: FastAPI + Uvicorn for Backend Services

- **Decision**: Build the application backend using **FastAPI** on an **Uvicorn** ASGI server.
- **Reason**: High performance, native async support, automatic OpenAPI/Swagger documentation generation (`/docs`), seamless integration with Pydantic v2 data validation, and minimal boilerplate.
- **Alternatives Considered**:
  - *Flask*: Synchronous by default, lacks automatic Pydantic schema validation.
  - *Django*: Excessive boilerplate, heavy ORM overhead unnecessary for a microservice.
- **Current Status**: **Active & Implemented** in `backend/main.py`.

---

### Decision 9: Bilingual English and Marathi (`मराठी`) Language Scope

- **Decision**: Provide native, first-class support for English and Marathi across all UI elements, guidance text, and audio voice synthesis.
- **Reason**: Maharashtra is a major agricultural hub in India for tomato, potato, and bell pepper cultivation. Marathi is the primary working language of regional smallholders who benefit most from voice-based accessibility.
- **Alternatives Considered**:
  - *English-Only*: Excludes non-English speaking farmers who are the primary beneficiaries.
  - *10+ Indian Languages Simultaneously*: Unmanageable prompt complexity and validation burden for an undergraduate engineering scope.
- **Current Status**: **Active & Implemented** in `frontend/app.js` and `src/utils/voice_service.py`.

---

### Decision 10: Monolithic Client-Server Architecture over Multi-Agent Frameworks

- **Decision**: Implement a clean, single-process FastAPI service with modular subpackages rather than a complex multi-agent orchestration framework (such as CrewAI, AutoGen, or LangGraph).
- **Reason**: A deterministic linear pipeline (Upload -> Quality -> Predict -> RAG -> Weather -> Gemini -> Response) executes in under 2 seconds. Multi-agent negotiation adds non-deterministic latency (15-45s), high token costs, and unnecessary architectural complexity for this problem.
- **Alternatives Considered**:
  - *Multi-Agent Consensus System*: High latency, high API costs, debugging complexity, and unneeded overhead for a single-leaf diagnostic task.
- **Current Status**: **Active Decision Preserved**.

---

### Decision 11: Explicit Confidence Reporting & Low-Confidence Warning Threshold ($	au = 0.60$)

- **Decision**: Surface both the model confidence percentage and a clear warning banner when confidence drops below 60%.
- **Reason**: Agricultural safety requires transparency. If a leaf is photographed poorly or symptoms are ambiguous, the farmer must be cautioned to consult a local Krishi Vigyan Kendra (KVK) officer rather than blindly applying chemical treatments.
- **Alternatives Considered**:
  - *Hiding Confidence*: Unsafe; gives equal weight to uncertain guesses.
  - *Strict Hard Rejection on Confidence < 0.60*: Frustrates users; soft warning with top-3 alternative possibilities is more helpful.
- **Current Status**: **Active & Implemented** in `src/prediction/predict.py` and `frontend/app.js`.

---

### Decision 12: Dual Weather API Strategy (OpenWeatherMap + Open-Meteo Fallback)

- **Decision**: Implement OpenWeatherMap as the primary provider with automatic, zero-configuration failover to the public Open-Meteo REST API.
- **Reason**: Guarantees that weather risk calculations never fail, even if the user has no OpenWeather API key or exceeds the free tier rate limit.
- **Alternatives Considered**:
  - *OpenWeatherMap Only*: Breaks weather context if the user's API key is missing or invalid.
  - *Static Climatological Data*: Fails to reflect current real-time humidity or rainfall events.
- **Current Status**: **Active & Implemented** in `src/weather/weather_service.py`.
