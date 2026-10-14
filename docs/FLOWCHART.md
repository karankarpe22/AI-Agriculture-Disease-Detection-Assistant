# 📊 AI Agriculture Assistant — System Flowcharts

This document provides visual flowcharts covering the end-to-end execution pipeline, decision gates, parallel processing subsystems, and offline fallback mechanisms.

---

## 1. Master System Flowchart

```mermaid
flowchart TD
    Start([Farmer Accesses Application]) --> InputChoice{Choose Mode}

    %% Branch A: Leaf Diagnosis
    InputChoice -->|Mode 1: Leaf Diagnosis| Upload[Upload Leaf Image + Optional Question + City]
    
    %% Branch B: Direct Q&A
    InputChoice -->|Mode 2: Ask Agronomist| DirectQ[Enter Farming Question + Select Crop + City]

    %% Quality Check Gate
    Upload --> QCheck{Image Quality Check<br/>src/preprocessing/quality_check.py}
    
    QCheck -->|Resolution < 100x100| RejectLowRes[Reject: Resolution Too Small]
    QCheck -->|Brightness < 30| RejectDark[Reject: Image Too Dark / Night]
    QCheck -->|Brightness > 230| RejectBright[Reject: Image Overexposed / Glare]
    QCheck -->|Laplacian Var < 35.0| RejectBlur[Reject: Image Out of Focus / Blurry]
    
    RejectLowRes --> ReUploadPrompt[Prompt Farmer: Steady Camera & Capture in Good Daylight]
    RejectDark --> ReUploadPrompt
    RejectBright --> ReUploadPrompt
    RejectBlur --> ReUploadPrompt
    ReUploadPrompt --> Upload

    %% Valid Image Processing
    QCheck -->|Image Quality Valid| Normalize[Tensor Normalization 224x224<br/>ImageNet Z-Score Standardization]

    Normalize --> MobileNet[MobileNetV3-Large Classifier<br/>src/prediction/predict.py]
    
    MobileNet --> Softmax[Compute Softmax Class Probabilities<br/>13 Agricultural Disease Classes]
    
    Softmax --> ConfCheck{Confidence >= 60%?}
    ConfCheck -->|Yes: High Confidence| ConfFlag[Status: Verified Diagnosis]
    ConfCheck -->|No: Low Confidence| WarnFlag[Flag: Low Confidence Warning<br/>Recommend KVK Extension Verification]

    MobileNet --> GradCAM[Grad-CAM Saliency Hook<br/>Compute Gradients on features-1<br/>Generate Heatmap Overlay]

    %% Parallel Context Ingestion
    Softmax --> JoinContext
    DirectQ --> JoinContext
    
    subgraph ParallelServices ["Parallel Context Ingestion"]
        JoinContext --> WeatherTask[Fetch Weather & Microclimate Risk<br/>Open-Meteo API / OpenWeatherMap]
        JoinContext --> RAGTask[FAISS Semantic Retrieval<br/>Top-3 ICAR Research Chunks]
    end

    WeatherTask --> Synthesize
    RAGTask --> Synthesize
    ConfFlag --> Synthesize
    WarnFlag --> Synthesize

    %% Generative Advisory with Fallback
    subgraph ReasoningLayer ["Advisory Synthesis & Fallback Cascade"]
        Synthesize[Aggregate Context: Diagnosis + Evidence + Microclimate + Question]
        Synthesize --> CallGemini{Call Gemini Multi-Model Cascade}
        
        CallGemini -->|HTTP 200: Success| LLMResponse[Gemini Model Generation<br/>gemini-flash-lite / gemini-3.6-flash]
        
        CallGemini -->|HTTP 429 Quota or 503 Outage| NextModel{Try Next Model in Cascade?}
        NextModel -->|Model Available| CallGemini
        NextModel -->|All Models Exhausted / Offline| OfflineEngine[Smart Offline Semantic Intent Engine<br/>Regex Parser: where, what, why, symptoms, neem, spray]
    end

    LLMResponse --> SchemaValidation[Enforce Strict JSON Schema<br/>direct_answer, management, prevention, precautions]
    OfflineEngine --> SchemaValidation

    %% Output Presentation Layer
    subgraph OutputLayer ["Farmer Output & Accessibility"]
        SchemaValidation --> DisplayUI[Render Clean UI: Disease Tag, Meter, Evidence, Formatted Advisory]
        GradCAM --> DisplayUI
        
        DisplayUI --> TTSCheck{Farmer Clicks Listen?}
        TTSCheck -->|Yes| AudioCache{Audio Cached in Disk?<br/>outputs/audio_cache/MD5.mp3}
        
        AudioCache -->|Cache Hit| FastAudio[Load Audio < 2ms]
        AudioCache -->|Cache Miss| AzureTTS[Azure Neural TTS: edge-tts<br/>en-IN-Neerja / mr-IN-Aarohi<br/>Normalize °C, g/L, WP @ 2.5]
        AzureTTS --> SaveCache[Save MP3 to Disk Cache]
        SaveCache --> FastAudio
        
        FastAudio --> PlayAudio([Play High-Fidelity Audio to Farmer])
        TTSCheck -->|No| Done([Session Ready for Follow-Up Question])
    end
```

---

## 2. Image Quality Gate Decision Flowchart

```mermaid
flowchart TD
    ImgIn([Input Leaf Photo]) --> Read[Decode Image Bytes to RGB]
    
    Read --> SizeGate{Width >= 100 px<br/>AND<br/>Height >= 100 px?}
    SizeGate -->|No| ErrSize[Return error: low_resolution]
    
    SizeGate -->|Yes| Gray[Convert to Grayscale Luma]
    Gray --> BrightCalc[Calculate Mean Pixel Brightness: μ]
    
    BrightCalc --> DarkGate{μ < 30.0?}
    DarkGate -->|Yes| ErrDark[Return error: too_dark / underexposed]
    
    DarkGate -->|No| BrightGate{μ > 230.0?}
    BrightGate -->|Yes| ErrBright[Return error: too_bright / flash glare]
    
    BrightGate -->|No| BlurCalc[Convolve with 3x3 Laplacian Kernel<br/>Calculate Variance: Var∇²I]
    
    BlurCalc --> BlurGate{Var >= 35.0?}
    BlurGate -->|No| ErrBlur[Return error: excessive_blur]
    
    BlurGate -->|Yes| Pass([Quality Check Passed: Proceed to MobileNetV3])
```

---

## 3. RAG & Gemini Multi-Model Cascade Flowchart

```mermaid
flowchart TD
    QueryIn([Diagnosed Disease + Farmer Query]) --> RAGFilter[Filter ICAR Knowledge Base by Crop & Disease]
    
    RAGFilter --> Embed[Generate 384-dim Dense Vector<br/>paraphrase-multilingual-MiniLM-L12-v2]
    
    Embed --> FAISSSearch[FAISS IndexFlatL2 Cosine Search]
    FAISSSearch --> Chunks[Extract Top-3 Grounded Evidence Chunks]
    
    Chunks --> PromptBuilder[Construct Grounded Prompt with Strict JSON Schema]
    
    PromptBuilder --> TryM1[Try Model 1: gemini-flash-lite-latest]
    
    TryM1 -->|Success 200| ValidJSON[Validate JSON Schema]
    TryM1 -->|Error 429 / 503| TryM2[Try Model 2: gemini-3.6-flash]
    
    TryM2 -->|Success 200| ValidJSON
    TryM2 -->|Error 429 / 503| TryM3[Try Model 3: gemini-flash-latest]
    
    TryM3 -->|Success 200| ValidJSON
    TryM3 -->|Error 429 / 503| TryM4[Try Model 4: gemini-2.5-flash]
    
    TryM4 -->|Success 200| ValidJSON
    TryM4 -->|Network Down / Quotas Exhausted| OfflineIntent[Activate Smart Offline Intent Matcher]
    
    OfflineIntent --> DetectIntent{Regex Intent Classification}
    DetectIntent -->|Where does it occur?| IntentWhere[Return Plant Organ: Lower foliage, stems, tubers]
    DetectIntent -->|What is it?| IntentWhat[Return Pathogen Definition: Alternaria / Xanthomonas]
    DetectIntent -->|Why did it happen?| IntentWhy[Return Environmental Cause: Humidity + Spores]
    DetectIntent -->|Symptoms?| IntentSymp[Return Visual Cues: Target rings, chlorotic halos]
    DetectIntent -->|Sprays / Neem?| IntentSpray[Return ICAR Fungicide Dosage: Mancozeb 75% WP @ 2g/L]
    
    IntentWhere --> FormatOffline[Format Grounded Offline Advisory JSON]
    IntentWhat --> FormatOffline
    IntentWhy --> FormatOffline
    IntentSymp --> FormatOffline
    IntentSpray --> FormatOffline
    
    FormatOffline --> Output([Return Advisory to Farmer])
    ValidJSON --> Output
```

---

## 4. Text-to-Speech (TTS) Normalization & Caching Flowchart

```mermaid
flowchart TD
    InputText([Advisory Text in English or Marathi]) --> Hash[Compute MD5 Hash of Text + Language]
    
    Hash --> CheckDisk{File exists in<br/>outputs/audio_cache/hash.mp3?}
    
    CheckDisk -->|Yes: Cache Hit| ReadFile[Read Audio Bytes Directly from Disk]
    ReadFile --> StreamAudio([Stream MP3 Audio to Browser < 2ms])
    
    CheckDisk -->|No: Cache Miss| CleanText[Text Normalization Pipeline]
    
    CleanText --> Step1[Strip Markdown Formatting: *, #, •, bullet points]
    Step1 --> Step2[Expand Abbreviations: °C -> degrees Celsius, g/L -> grams per liter]
    Step2 --> Step3[Expand Chemical Notation: WP @ 2.5 -> wettable powder at 2.5]
    Step3 --> Step4[Expand Acronyms: ICAR -> I.C.A.R.]
    
    Step4 --> VoiceSelect{Language?}
    VoiceSelect -->|English| VoiceEN[Azure Voice: en-IN-NeerjaExpressiveNeural]
    VoiceSelect -->|Marathi| VoiceMR[Azure Voice: mr-IN-AarohiNeural]
    
    VoiceEN --> CallEdge[Invoke edge-tts Asynchronous Stream]
    VoiceMR --> CallEdge
    
    CallEdge -->|Success| SaveDisk[Write MP3 to outputs/audio_cache/hash.mp3]
    SaveDisk --> StreamAudio
    
    CallEdge -->|Failure / Network Timeout| FallbackBrowser[Client-Side Web Speech API Fallback]
    FallbackBrowser --> StreamAudio
```

---

## 5. ASCII System Flowchart (For Project Reports & PPT Slides)

```text
================================================================================================
                                AI AGRICULTURE ASSISTANT FLOWCHART
================================================================================================

 [ Farmer Uploads Leaf Image ]               [ Farmer Enters Direct Question ]
              │                                              │
              ▼                                              │
  ┌───────────────────────┐                                  │
  │  Image Quality Gate   │                                  │
  │ • Blur (Laplacian)    │──[Fail: Blur/Dark/Glare]──► [ Re-Prompt Farmer with Advice ]
  │ • Brightness (Exposure│                                  │
  │ • Resolution (>=100px)│                                  │
  └───────────┬───────────┘                                  │
              │ (Pass)                                       │
              ▼                                              │
  ┌───────────────────────┐                                  │
  │ MobileNetV3-Large CNN │                                  │
  │ • Softmax Probability │                                  │
  │ • Grad-CAM Heatmap    │                                  │
  └───────────┬───────────┘                                  │
              │                                              │
              ▼                                              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                 Parallel Context Ingestion Engine                      │
  │  • Real-Time Weather: Open-Meteo (Humidity, Temp, Infection Risk)      │
  │  • RAG Vector Retrieval: FAISS + ICAR Grounded Research Evidence Chunks│
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │               Google Gemini Multi-Model Cascade Layer                  │
  │  Primary: gemini-flash-lite  ──► Secondary: gemini-3.6-flash           │
  │  (Fallback on 429 Quota Exceeded / 503 Network Unavailable)            │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     [ Gemini Online ]                              [ Network / Quota Offline ]
  Enforce Strict JSON Schema                     Smart Semantic Intent Regex Engine
  (Direct Answer, Sprays, Safety)                (Where, What, Why, Symptoms, Neem)
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                         Farmer Results Panel                           │
  │  • Primary Disease & Calibrated Confidence Meter                       │
  │  • Grad-CAM Visual Heatmap Highlighting Foliar Lesions                 │
  │  • Direct Answer to Question + ICAR Management & Prevention            │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                  Human-Like Neural Speech Subsystem                    │
  │  • Disk Cache Lookup (<2ms)                                            │
  │  • Agricultural Text Normalizer (°C, g/L, WP @ 2.5)                    │
  │  • Microsoft Azure Neural Voice: Neerja (English) & Aarohi (Marathi)   │
  └────────────────────────────────────────────────────────────────────────┘
```
