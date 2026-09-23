# PRD.md — Product Requirements Document

> **Product Name**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Product Stage**: Production Baseline (Version 1.0.0)  
> **Document Status**: Approved & Audited  
> **Requirement Status Tags**: [IMPLEMENTED] | [PLANNED] | [OPTIONAL] | [FUTURE]  

---

## 1. Product Overview & Problem Statement

### 1.1 Overview
The **AI Agriculture Assistant** is an intelligent, evidence-grounded plant health diagnosis and agro-advisory system designed for Indian smallholder farmers. It accepts smartphone photographs of diseased crop leaves (Tomato, Potato, Bell Pepper), detects pathology using an ImageNet-pretrained **MobileNetV3-Large** neural network, evaluates local microclimatic risk factors via real-time weather data, retrieves authoritative cultural and biological management guidelines from an Indian Council of Agricultural Research (**ICAR**) knowledge base, and synthesizes structured, actionable recommendations in English and Marathi using **Google Gemini**.

### 1.2 Problem Statement
- **Diagnostic Delay & Misidentification**: Smallholder farmers often misdiagnose foliar fungal and bacterial infections (e.g., confusing Early Blight with Septoria Leaf Spot), leading to incorrect pesticide applications that waste capital and induce pathogen resistance.
- **Agricultural Extension Gap**: With an agricultural extension officer-to-farmer ratio exceeding 1:1,000 in India, farmers lack access to timely, personalized, and scientifically verified agronomic guidance.
- **LLM Hallucination Risk**: Generic multimodal AI models hallucinate dangerous chemical dosages when asked to inspect crop images directly.
- **Language & Literacy Barriers**: Rural farmers require voice-accessible, localized advisory in regional languages (such as Marathi) rather than complex English text reports.

---

## 2. Target Users & User Needs

| User Persona | Profile | Core Pain Points | Expected Value |
| :--- | :--- | :--- | :--- |
| **Smallholder Farmer (Ramesh)** | Cultivates 2–4 acres of tomato and potato in Maharashtra; smartphone user; prefers Marathi; limited technical literacy. | Noticeable foliar spotting; unsure whether to spray fungicide or pesticide; risk of losing entire seasonal harvest. | Uploads leaf photo; receives instant diagnosis and spoken Marathi advice on affordable, organic/chemical ICAR treatments. |
| **Krishi Vigyan Kendra (KVK) Extension Officer (Sunita)** | Field agronomist serving multiple rural blocks; conducts field surveys. | Overburdened with queries; needs quick preliminary screening of farmer samples with explainability. | Uses the web dashboard to review leaf Grad-CAM heatmaps, top-3 candidates, and ICAR references for field validation. |

---

## 3. Product Goals & Success Metrics

- **Diagnostic Latency**: Under 2.0 seconds end-to-end for combined perception, RAG, and weather synthesis. [IMPLEMENTED]
- **Diagnostic Accuracy**: Exceed 80% overall accuracy on unseen, class-stratified test images. [IMPLEMENTED: 81.89%]
- **Evidence Grounding**: Zero fabricated or unapproved chemical recommendations; 100% of treatment advice traceable to ICAR/TNAU compendiums. [IMPLEMENTED]
- **Zero-Cost Operation**: Run entirely within free cloud hosting tiers (Render Free Web Service, Gemini Free Tier, OpenWeather Free Tier) without billing or user charges. [IMPLEMENTED]

---

## 4. Functional Requirements

### 4.1 Image Upload & Quality Check
- **FR-1.1**: The system must allow users to upload crop leaf images via drag-and-drop or file selector. `[IMPLEMENTED]`
- **FR-1.2**: The system must enforce a 10MB maximum file size limit and validate MIME types (`image/jpeg`, `image/png`, `image/webp`). `[IMPLEMENTED]`
- **FR-1.3**: The system must inspect image resolution ($\ge 128 	imes 128$), brightness (mean luminance $40.0 - 230.0$), and blur (Laplacian variance $\ge 50.0$), rejecting invalid images before model inference. `[IMPLEMENTED]`

### 4.2 Disease Classification & Perception
- **FR-2.1**: The system must classify the leaf into one of 13 calibrated classes across Tomato, Potato, and Bell Pepper using MobileNetV3-Large. `[IMPLEMENTED]`
- **FR-2.2**: The system must report the winning class, crop name, disease name, and percentage confidence score. `[IMPLEMENTED]`
- **FR-2.3**: If confidence drops below 60%, the system must flag `is_low_confidence = True` and render a warning banner urging expert verification. `[IMPLEMENTED]`
- **FR-2.4**: The system must provide the top-3 candidate predictions with relative probabilities in a collapsible details view. `[IMPLEMENTED]`
- **FR-2.5**: The system must generate a Grad-CAM visual attention heatmap highlighting regions influencing model prediction upon user request. `[IMPLEMENTED]`

### 4.3 Environmental Context & Weather Integration
- **FR-3.1**: The system must accept a farmer location string (e.g., "Pune", "Nashik") and retrieve live temperature, humidity, and rainfall. `[IMPLEMENTED]`
- **FR-3.2**: The system must calculate a microclimatic disease risk index correlating current weather with pathogen spread. `[IMPLEMENTED]`
- **FR-3.3**: The system must automatically fail over to Open-Meteo API if OpenWeatherMap is unavailable or throttled. `[IMPLEMENTED]`

### 4.4 Agricultural Knowledge Retrieval (RAG)
- **FR-4.1**: The system must retrieve peer-reviewed ICAR / TNAU evidence chunks matching the diagnosed crop, disease, and farmer question. `[IMPLEMENTED]`
- **FR-4.2**: The retrieval engine must operate with zero memory overhead, avoiding resident transformer models in RAM. `[IMPLEMENTED]`
- **FR-4.3**: The system must display source citations and document references for transparency. `[IMPLEMENTED]`

### 4.5 Generative Advisory & Multilingual Synthesis
- **FR-4.6**: The system must prompt Google Gemini (`gemini-2.5-flash`) using strict ground truth from MobileNetV3 and ICAR chunks. `[IMPLEMENTED]`
- **FR-4.7**: Gemini must structure advice into Explanation, Weather Impact, Management/Treatment, and Prevention. `[IMPLEMENTED]`
- **FR-4.8**: The system must support native English and Marathi (`मराठी`) language modes. `[IMPLEMENTED]`
- **FR-4.9**: If Gemini API is unreachable, the system must trigger a deterministic ICAR rule-based fallback generator. `[IMPLEMENTED]`

### 4.6 Voice Accessibility
- **FR-5.1**: The system must provide speech-to-text (STT) voice input for farmer queries using browser Web Speech API. `[IMPLEMENTED]`
- **FR-5.2**: The system must provide text-to-speech (TTS) streaming audio in English and Marathi using gTTS. `[IMPLEMENTED]`
- **FR-5.3**: Audio files must be cached locally to prevent redundant synthesis requests. `[IMPLEMENTED]`

---

## 5. Non-Functional Requirements

- **NFR-1 (Performance)**: CPU vision inference latency $\le 30 	ext{ ms}$ per image; total end-to-end pipeline latency $\le 2.5 	ext{ seconds}$. `[IMPLEMENTED]`
- **NFR-2 (Memory Footprint)**: Resident backend RAM must remain strictly below 400 MB to operate reliably on 512 MB free hosting tiers. `[IMPLEMENTED]`
- **NFR-3 (Security)**: API keys must remain strictly in backend environment variables, never leaked to the client or public Git repositories. `[IMPLEMENTED]`
- **NFR-4 (Reliability)**: The service must degrade gracefully during third-party API outages with zero 500 error crashes. `[IMPLEMENTED]`
- **NFR-5 (Usability)**: Responsive glassmorphic UI adhering to modern mobile accessibility guidelines. `[IMPLEMENTED]`

---

## 6. Scope Boundaries & Roadmap

### 6.1 Currently Implemented [IMPLEMENTED]
- 13-class MobileNetV3-Large transfer-learning classifier with 81.89% test accuracy.
- Image quality check layer (blur, darkness, brightness, resolution).
- 62-chunk ICAR-grounded knowledge base and zero-RAM RAG retrieval.
- Dual weather integration (OpenWeatherMap + Open-Meteo).
- Google Gemini 2.5 Flash generative advisory with rule-based fallback.
- Bilingual English and Marathi interface with gTTS voice synthesis and STT voice input.
- Grad-CAM visual attention heatmaps.
- FastAPI backend with 10 endpoints and single-port static frontend serving.

### 6.2 Planned Enhancements [PLANNED]
- Offline progressive web app (PWA) manifest for home screen installation on Android devices.
- Multi-crop expansion to include Cotton, Soybean, and Rice.
- Leaf symptom severity quantification (estimating percentage leaf area infected).

### 6.3 Optional Features [OPTIONAL]
- Hindi (`हिन्दी`) language support expansion.
- Integration with local Krishi Vigyan Kendra (KVK) WhatsApp direct contact directory.
- Downloadable PDF diagnostic health card.

### 6.4 Out of Scope [FUTURE / EXCLUDED]
- Autonomous pesticide dispensing or drone flight controller integration.
- Direct chemical e-commerce sales or payment gateway integration.
- Soil sensor IoT hardware fabrication.
