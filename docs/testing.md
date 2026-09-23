# TESTING.md — Testing Strategy & Comprehensive Test Matrix

> **Project**: AI Agriculture Assistant (कृषी सहाय्यक)  
> **Status**: Verified & Operational  
> **Test Framework**: `pytest 9.1+` with `fastapi.testclient.TestClient`  
> **Total Automated Tests**: 24  
> **Pass Rate**: 100% (24 Passed, 0 Failed)  

---

## 1. Testing Philosophy & Strategy

The project employs a multi-tiered testing methodology to guarantee that each decoupled layer operates reliably in isolation and as a cohesive whole:
1. **Unit Testing**: Tests core mathematical calculations, string parsing, image validation formulas, and isolated service classes.
2. **Component Testing**: Verifies PyTorch MobileNetV3 layer structure, ImageNet transform dimensions, RAG chunk metadata extraction, and weather fallback switching.
3. **Integration Testing**: Verifies composite pipelines (`/predict`, `/analyze`, `/ask`) with multi-part form payloads, image streaming, and multilingual responses.
4. **End-to-End Testing**: Validates end-to-end user workflows from raw image upload to speech synthesis and Grad-CAM visualization.
5. **Edge Case & Adversarial Testing**: Validates system resilience against pitch-black images, overexposed whiteouts, microscopic crops, and non-image files.

---

## 2. Automated Test Matrix (24 Tests)

All 24 automated tests below are located in the `tests/` directory and were executed against the active codebase:

| Test ID | Test File | Component | Test Function | Expected Result | Actual Result | Status |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **TC-01** | `test_api.py` | FastAPI Backend | `test_health_endpoint` | HTTP 200, status `healthy`, 13 classes listed | HTTP 200, `healthy`, 13 classes verified | **PASS** |
| **TC-02** | `test_api.py` | Weather Endpoint | `test_weather_endpoint` | HTTP 200, contains temperature, humidity, risk | HTTP 200, valid weather payload returned | **PASS** |
| **TC-03** | `test_api.py` | Prediction API | `test_predict_endpoint_valid_image` | HTTP 200, `success: True`, crop & disease returned | HTTP 200, valid prediction returned | **PASS** |
| **TC-04** | `test_api.py` | Quality Guard | `test_predict_rejects_dark_image` | HTTP 200, `success: False`, quality warning | HTTP 200, rejected with dark image warning | **PASS** |
| **TC-05** | `test_api.py` | Voice Audio API | `test_tts_endpoint` | HTTP 200, `Content-Type: audio/mpeg`, non-empty bytes | HTTP 200, streaming MP3 binary returned | **PASS** |
| **TC-06** | `test_dataset_tools.py` | Data Tools | `test_inspection_detects_classes_and_corruption` | Identifies valid files and flags corrupted image byte streams | Correctly counts images and detects corrupt dummy files | **PASS** |
| **TC-07** | `test_dataset_tools.py` | Data Splitting | `test_split_class_counts` | Splits dataset according to 70/15/15 target ratios | Correctly partitions train, validation, and test splits | **PASS** |
| **TC-08** | `test_gemini.py` | Gemini Guidance | `test_guidance_generation_structure` | Returns structured dict with explanation, management, prevention | Valid structured dictionary returned | **PASS** |
| **TC-09** | `test_gemini.py` | Marathi Localization | `test_marathi_guidance_generation` | Returns guidance in Marathi with non-empty management steps | Returns Marathi Devanagari text with valid guidance list | **PASS** |
| **TC-10** | `test_integration.py` | Full Pipeline (EN) | `test_full_analyze_pipeline_english` | HTTP 200, complete diagnosis, weather, evidence, and guidance in English | All pipeline stages execute and return valid response | **PASS** |
| **TC-11** | `test_integration.py` | Full Pipeline (MR) | `test_full_analyze_pipeline_marathi` | HTTP 200, complete diagnosis, weather, evidence, and guidance in Marathi | All pipeline stages execute and return Marathi guidance | **PASS** |
| **TC-12** | `test_model.py` | MobileNetV3 Head | `test_mobilenet_head_is_replaced_and_backbone_frozen` | Classifier replaced with 13 outputs, backbone requires_grad=False | Head output matches 13 classes, backbone frozen | **PASS** |
| **TC-13** | `test_prediction.py` | Label Formatter | `test_format_class_name` | Parses raw folder string into clean `(Crop, Disease)` pair | Formatted correctly for all 13 class labels | **PASS** |
| **TC-14** | `test_prediction.py` | Predictor Core | `test_predictor_loads_and_predicts` | Loads `mobilenetv3_best.pth`, outputs probabilities summing to ~1.0 | Model loads cleanly, predicts valid class distribution | **PASS** |
| **TC-15** | `test_prediction.py` | Explainability | `test_predictor_gradcam_generation` | Generates PIL Image heatmap overlay of shape (224, 224, 3) | Valid RGB heatmap PIL image generated | **PASS** |
| **TC-16** | `test_quality_check.py` | Quality Check | `test_valid_image_passes_quality_check` | Clear leaf image returns `is_valid: True` | Clean leaf passes all checks | **PASS** |
| **TC-17** | `test_quality_check.py` | Resolution Check | `test_too_small_resolution_fails` | Image < 128x128 pixels returns `is_valid: False` | Rejects 64x64 dummy image with resolution warning | **PASS** |
| **TC-18** | `test_quality_check.py` | Exposure Check (Dark) | `test_extremely_dark_image_fails` | Mean brightness < 40 returns `is_valid: False` | Rejects black dummy image with dark warning | **PASS** |
| **TC-19** | `test_quality_check.py` | Exposure Check (Bright)| `test_extremely_bright_image_fails` | Mean brightness > 230 returns `is_valid: False` | Rejects overexposed dummy image with bright warning | **PASS** |
| **TC-20** | `test_quality_check.py` | Sharpness Check | `test_excessively_blurry_image_fails` | Laplacian variance < 50 returns `is_valid: False` | Rejects uniform gray image with blur warning | **PASS** |
| **TC-21** | `test_rag.py` | RAG Retrieval | `test_rag_retrieves_relevant_evidence` | Retrieves top-3 relevant ICAR chunks for crop & disease | Returns top-3 chunks matching crop and disease | **PASS** |
| **TC-22** | `test_rag.py` | Citation Formatter | `test_rag_sources_summary` | Returns deduplicated list of source authorities | Deduplicated list of ICAR authorities returned | **PASS** |
| **TC-23** | `test_weather.py` | Weather Fetcher | `test_weather_service_fetches_real_data` | Successfully retrieves live meteorological data | Live weather payload fetched with valid metrics | **PASS** |
| **TC-24** | `test_weather.py` | Risk Heuristic | `test_disease_risk_analysis_logic` | High humidity (>80%) flags elevated pathogen pressure | High humidity correctly triggers fungal risk alert | **PASS** |

---

## 3. How to Execute the Test Suite

Run all tests from the repository root:
```bash
python -m pytest tests/ -v
```

Run a specific test module:
```bash
python -m pytest tests/test_prediction.py -v
python -m pytest tests/test_api.py -v
```

Run with execution timing:
```bash
python -m pytest tests/ --durations=5
```

---

## 4. Manual Verification & Smoke Test Procedures

### 4.1 Server Launch Verification
1. Execute `run.bat` or `run.ps1`.
2. Observe console logs:
   ```text
   INFO:     Started server process
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://127.0.0.1:8000
   ```
3. Open browser at `http://localhost:8000`. Confirm the header badge reads `MobileNetV3-Large Ready (13 classes)`.

### 4.2 End-to-End Diagnostic Flow Check
1. Drag and drop sample image `data/samples/sample_tomato_early_blight.jpg`.
2. Enter location `Nashik` and select `English`.
3. Check `Generate Grad-CAM Visual Attention Heatmap`.
4. Click **Analyze Crop Health**.
5. **Expected Results**:
   - Disease card displays: `Tomato Early Blight` with confidence score.
   - Weather widget displays live Nashik temperature, humidity, and microclimate risk.
   - Evidence card renders ICAR recommendations (e.g., Mancozeb dosage).
   - Grad-CAM heatmap renders showing attention on concentric leaf lesions.
   - "Listen to Advisory" plays clear spoken MP3 audio.

---

## 5. Field Robustness & Real-World Considerations

- **Out-of-Distribution Rejection**: Unrelated images (e.g., human faces, vehicles, plain soil) produce low confidence scores (< 60%), properly triggering the yellow uncertainty warning banner.
- **Network Outage Resilience**: If the internet connection drops during an analysis, the backend gracefully falls back to local ICAR guidelines, ensuring farmers never face an unhandled application crash.
