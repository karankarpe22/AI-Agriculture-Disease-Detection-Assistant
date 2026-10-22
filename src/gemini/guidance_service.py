"""Gemini LLM Guidance Service for Evidence-Grounded Agricultural Advisory.

Integrates with the official Google GenAI SDK (`google-genai`), grounding LLM responses
strictly on verified ICAR/TNAU RAG context, live weather microclimates, and model outputs.
Provides dual-language (English and Marathi) structured guidance so that switching
languages on the frontend happens instantaneously without losing analysis context.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("src.gemini.guidance_service")

# Candidate models in priority order
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]


class GeminiGuidanceService:
    """Service wrapper for Google Gemini with multi-model fallback and strict RAG grounding."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model_name or "gemini-2.5-flash"
        self._client = None

    def _get_client(self):
        """Lazy initialization of the official Google GenAI Client."""
        if self._client is None and self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai Client: {e}")
                self._client = None
        return self._client

    def generate_guidance(
        self,
        crop: str,
        disease: str,
        confidence: float,
        weather: dict[str, Any],
        retrieved_evidence: list[dict[str, Any]],
        farmer_question: str | None = None,
        language: str = "english",
        is_low_confidence: bool = False,
    ) -> dict[str, Any]:
        """Generate structured agricultural guidance with multi-model fallback and dual EN/MR outputs."""
        lang = (language or "english").lower().strip()
        is_marathi = lang in ["marathi", "mr", "मराठी"]

        # If Gemini API key is available, call official Google GenAI SDK across candidate models
        client = self._get_client()
        if client:
            models_to_try = [self.model_name] + [m for m in DEFAULT_MODELS if m != self.model_name]
            for m in models_to_try:
                try:
                    guidance = self._call_gemini(
                        client=client,
                        model=m,
                        crop=crop,
                        disease=disease,
                        confidence=confidence,
                        weather=weather,
                        evidence=retrieved_evidence,
                        question=farmer_question,
                        is_marathi=is_marathi,
                        is_low_confidence=is_low_confidence,
                    )
                    if guidance:
                        self._ensure_bilingual_completeness(
                            guidance=guidance,
                            crop=crop,
                            disease=disease,
                            confidence=confidence,
                            weather=weather,
                            evidence=retrieved_evidence,
                            question=farmer_question,
                            is_marathi=is_marathi,
                            is_low_confidence=is_low_confidence,
                        )
                        return guidance
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    is_busy = "503" in err_str or "UNAVAILABLE" in err_str
                    if is_rate_limit or is_busy:
                        logger.warning(f"Gemini model '{m}' rate-limited or busy. Trying next model...")
                        continue
                    else:
                        logger.warning(f"Gemini call on '{m}' failed: {e}. Trying next model...")
                        continue

            logger.info("All Gemini candidate models were exhausted or unavailable. Falling back to grounded RAG formatter.")

        # Robust Evidence-Grounded Fallback Engine
        return self._fallback_grounded_guidance(
            crop=crop,
            disease=disease,
            confidence=confidence,
            weather=weather,
            evidence=retrieved_evidence,
            question=farmer_question,
            is_marathi=is_marathi,
            is_low_confidence=is_low_confidence,
        )

    def _ensure_bilingual_completeness(
        self,
        guidance: dict[str, Any],
        crop: str,
        disease: str,
        confidence: float,
        weather: dict[str, Any],
        evidence: list[dict[str, Any]],
        question: str | None,
        is_marathi: bool,
        is_low_confidence: bool,
    ) -> None:
        """Guarantee that both 'en' and 'mr' sub-dictionaries exist in guidance result."""
        fallback = self._fallback_grounded_guidance(
            crop=crop,
            disease=disease,
            confidence=confidence,
            weather=weather,
            evidence=evidence,
            question=question,
            is_marathi=is_marathi,
            is_low_confidence=is_low_confidence,
        )

        en_block = guidance.get("en")
        if not isinstance(en_block, dict) or not en_block.get("explanation"):
            if not is_marathi and guidance.get("explanation"):
                en_block = {
                    "direct_answer": guidance.get("direct_answer", ""),
                    "explanation": guidance.get("explanation", ""),
                    "weather_interpretation": guidance.get("weather_interpretation", ""),
                    "management_guidance": guidance.get("management_guidance", ""),
                    "prevention": guidance.get("prevention", ""),
                    "precautions": guidance.get("precautions", ""),
                    "expert_advisory": guidance.get("expert_advisory", ""),
                }
            else:
                en_block = fallback.get("en", {})
        guidance["en"] = en_block

        mr_block = guidance.get("mr")
        if not isinstance(mr_block, dict) or not mr_block.get("explanation"):
            if is_marathi and guidance.get("explanation"):
                mr_block = {
                    "direct_answer": guidance.get("direct_answer", ""),
                    "explanation": guidance.get("explanation", ""),
                    "weather_interpretation": guidance.get("weather_interpretation", ""),
                    "management_guidance": guidance.get("management_guidance", ""),
                    "prevention": guidance.get("prevention", ""),
                    "precautions": guidance.get("precautions", ""),
                    "expert_advisory": guidance.get("expert_advisory", ""),
                }
            else:
                mr_block = fallback.get("mr", {})
        guidance["mr"] = mr_block

        guidance["direct_answer_en"] = en_block.get("direct_answer", guidance.get("direct_answer", ""))
        guidance["direct_answer_mr"] = mr_block.get("direct_answer", "")

    def _build_prompt(
        self,
        crop: str,
        disease: str,
        confidence: float,
        weather: dict[str, Any],
        evidence: list[dict[str, Any]],
        question: str | None,
        is_marathi: bool,
        is_low_confidence: bool,
    ) -> str:
        evidence_text = "\n\n".join([
            f"[Source: {item.get('source', 'Agricultural Extension')}, Title: {item.get('title', 'Advisory')}]\n"
            f"Section: {item.get('heading', 'General')}\n{item.get('content', '')}"
            for item in evidence
        ])

        weather_summary = (
            f"Location: {weather.get('location', 'Local Farm')}, "
            f"Temp: {weather.get('temperature_c', 'N/A')}°C, "
            f"Humidity: {weather.get('humidity_percentage', 'N/A')}%, "
            f"Conditions: {weather.get('condition', 'N/A')}. "
            f"Risk note: {weather.get('risk_analysis', 'N/A')}"
        )

        has_specific_question = bool(question and question.strip())

        if has_specific_question:
            qa_instruction = f"""CRITICAL Q&A INSTRUCTION:
The farmer asked this SPECIFIC QUESTION: "{question.strip()}"
You MUST provide a direct, comprehensive, empathetic, and highly actionable answer addressing this EXACT question in the "direct_answer" field of BOTH the "en" and "mr" blocks.
- If the farmer asks "how to prevent", "what care should be taken", or "how to control", specify exact varieties, prophylactic earthing up, sanitation, and chemical/organic sprays with dosage (e.g. Mancozeb 75% WP @ 2.5 g/L).
- If the farmer asks "where does this occur", explain precisely which plant organs (lower foliage, stems, tubers, fruits) and field conditions.
- If the farmer asks about organic or neem spray, provide exact dilution ratios (e.g. 5ml neem oil / L water) and application intervals.
- Make the "direct_answer" the centerpiece of your advisory in BOTH English ("en") and authentic Marathi ("mr")."""
        else:
            qa_instruction = """No specific question was submitted. In the "direct_answer" field of both "en" and "mr", provide a clear 2-3 sentence executive summary of the most urgent immediate next step the farmer must take today to protect their crop."""

        is_general_inquiry = "general" in (crop or "").lower() or "general" in (disease or "").lower()

        prompt = f"""You are an expert AI Agriculture Assistant advising an Indian farmer in Maharashtra.
Your role is to provide safe, actionable, and grounded agricultural guidance in BOTH English and Marathi (मराठी).

CRITICAL CONSTRAINTS:
1. {'Do NOT claim you diagnosed the leaf image. The disease classification was performed separately by a MobileNetV3 computer vision model.' if not is_general_inquiry else 'You are advising on a general crop management query without an image. Ground your advice in ICAR agronomic principles.'}
2. Ground all biological, cultural, and chemical advice strictly in the provided Agricultural Evidence below. Do NOT invent unsupported pesticide dosages or dangerous chemical mixtures.
3. Consider the provided Weather Context when discussing disease spread or spray timing.
4. Output MUST be valid JSON containing TWO top-level objects: "en" (English guidance) and "mr" (Marathi मार्गदर्शन).
{'5. The disease prediction has LOW CONFIDENCE. Emphasize uncertainty and advise the farmer to consult their local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying chemicals.' if is_low_confidence else '5. Always remind the farmer to confirm with local agricultural officers before large-scale chemical applications.'}

{qa_instruction}

INPUT DATA:
- Classified Crop: {crop or 'General Solanaceae / Agriculture'}
- Classified Disease / Topic: {disease or 'Agronomic Inquiry'}
- Model Confidence: {confidence * 100:.1f}% ({'LOW CONFIDENCE' if is_low_confidence else 'NORMAL'})
- Weather Context: {weather_summary}
- Farmer's Question: {question or 'What should I do to treat and protect my crop?'}

RETRIEVED AGRICULTURAL EVIDENCE:
{evidence_text if evidence_text else 'Standard ICAR Horticultural Practices'}

Please return your response in JSON format with exactly TWO top-level objects: "en" and "mr":
{{
  "en": {{
    "direct_answer": "Comprehensive, empathetic, and actionable answer in English to the farmer's question.",
    "explanation": "Simple, clear explanation in English of what this condition is and why it occurred.",
    "weather_interpretation": "How current temperature and humidity impact this disease or spray timing in English.",
    "management_guidance": "Clear bulleted cultural, organic, and chemical actions with dosage in English.",
    "prevention": "Practical measures in English to protect next crop cycle or healthy foliage.",
    "precautions": "Safety measures, PPE, and pre-harvest interval in English.",
    "expert_advisory": "Recommendation in English to consult local KVK or extension officer."
  }},
  "mr": {{
    "direct_answer": "शेतकऱ्याच्या प्रश्नाचे थेट, आदरयुक्त आणि परिपूर्ण मराठीत उत्तर.",
    "explanation": "हा रोग काय आहे आणि कशामुळे झाला याचे सोप्या मराठीत स्पष्टीकरण.",
    "weather_interpretation": "सध्याचे तापमान आणि हवेतील आर्द्रता या रोगावर कसा परिणाम करते याचे मराठीत विश्लेषण.",
    "management_guidance": "आयसीएआर शिफारशीनुसार उपाययोजना, सेंद्रिय व रासायनिक फवारणीचे प्रमाण मराठीत.",
    "prevention": "पुढील हंगामासाठी किंवा निरोगी झाडांसाठी दीर्घकालीन प्रतिबंधात्मक उपाय मराठीत.",
    "precautions": "फवारणी करताना घ्यावयाची काळजी आणि तोडणीपूर्वीचा प्रतीक्षा काळ मराठीत.",
    "expert_advisory": "जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा कृषी अधिकाऱ्यांचा सल्ला घेण्याची शिफारस."
  }}
}}

Return ONLY valid JSON. Do not include markdown ticks like ```json."""
        return prompt

    def _call_gemini(
        self,
        client: Any,
        model: str,
        crop: str,
        disease: str,
        confidence: float,
        weather: dict[str, Any],
        evidence: list[dict[str, Any]],
        question: str | None,
        is_marathi: bool,
        is_low_confidence: bool,
    ) -> dict[str, Any] | None:
        prompt = self._build_prompt(
            crop, disease, confidence, weather, evidence, question, is_marathi, is_low_confidence
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        raw_text = response.text.strip()
        clean_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"^```\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text).strip()

        try:
            parsed = json.loads(clean_text)
            en_block = parsed.get("en") if isinstance(parsed.get("en"), dict) else {}
            mr_block = parsed.get("mr") if isinstance(parsed.get("mr"), dict) else {}

            if not en_block and not mr_block:
                if is_marathi:
                    mr_block = dict(parsed)
                else:
                    en_block = dict(parsed)

            target_block = mr_block if (is_marathi and mr_block) else (en_block or parsed)
            guidance = dict(target_block)
            guidance["en"] = en_block or guidance
            guidance["mr"] = mr_block or {}
            guidance["language"] = "marathi" if is_marathi else "english"
            guidance["powered_by"] = f"Google Gemini ({model})"

            if not guidance.get("direct_answer"):
                guidance["direct_answer"] = (
                    mr_block.get("direct_answer") if is_marathi else en_block.get("direct_answer", "")
                )

            return guidance

        except Exception as e:
            logger.warning(f"Failed to parse Gemini JSON output: {e}. Raw was: {raw_text[:200]}")
            return None

    def _fallback_grounded_guidance(
        self,
        crop: str,
        disease: str,
        confidence: float,
        weather: dict[str, Any],
        evidence: list[dict[str, Any]],
        question: str | None,
        is_marathi: bool,
        is_low_confidence: bool,
    ) -> dict[str, Any]:
        """Evidence-grounded fallback synthesized directly from RAG documents with intelligent intent matching in both languages."""
        mgmt_points = []
        symptoms_points = []
        for item in evidence:
            content = item.get("content", "")
            heading = item.get("heading", "").lower()
            lines = [l.strip("- 1234567890.") for l in content.splitlines() if len(l.strip()) > 15]
            if "chemical" in heading or "management" in heading or "control" in heading:
                mgmt_points.extend(lines[:3])
            elif "symptom" in heading:
                symptoms_points.extend(lines[:2])

        weather_note = weather.get("risk_analysis", "Current weather conditions are within normal seasonal range.")

        # Intent detection
        q_clean = (question or "").strip()
        q_lower = q_clean.lower()

        is_care = any(k in q_lower for k in ["care", "prevent", "protect", "stop", "avoid", "काळजी", "प्रतिबंध", "कशी घ्यावी", "बचाव", "उपाय", "रोखायचे"])
        is_where = any(k in q_lower for k in ["where", "occur", "location", "place", "part", "foliage", "leaf", "कुठे", "कोणत्या भागावर", "स्थान"])
        is_what = any(k in q_lower for k in ["what", "meaning", "definition", "identify", "which disease", "काय", "काय आहे", "रोग कोणता"])
        is_why = any(k in q_lower for k in ["why", "cause", "reason", "pathogen", "का", "कशामुळे", "कारण"])
        is_symptom = any(k in q_lower for k in ["symptom", "sign", "look", "spot", "लक्षण", "दिसते कसे", "डाग"])
        is_when = any(k in q_lower for k in ["when", "season", "time", "month", "कधी", "केव्हा", "हंगाम", "वेळ"])
        is_neem = any(k in q_lower for k in ["neem", "organic", "bio", "कडुनिंब", "सेंद्रिय", "जैविक"])
        is_spray = any(k in q_lower for k in ["spray", "fungicide", "pesticide", "chemical", "medicine", "dose", "फवारणी", "औषध", "कीटकनाशक", "बुरशीनाशक"])
        is_water = any(k in q_lower for k in ["water", "irrigation", "moisture", "पाणी", "सिंचन", "ओलावा"])
        is_harvest = any(k in q_lower for k in ["harvest", "fruit", "market", "eat", "phi", "waiting", "काढणी", "फळ", "खाण्यायोग्य", "तोडणी"])

        crop_mr = {"Tomato": "टोमॅटो", "Potato": "बटाटा", "Bell pepper": "ढोबळी मिरची", "Pepper, bell": "ढोबळी मिरची"}.get(crop, crop)
        disease_mr = {
            "Early blight": "लवकर येणारा करपा (Early Blight)",
            "Late blight": "उशिरा येणारा करपा (Late Blight)",
            "Bacterial spot": "जिवाणूजन्य ठिपके (Bacterial Spot)",
            "Leaf Mold": "पानावरील बुरशी (Leaf Mold)",
            "Septoria leaf spot": "सेप्टोरिया पानावरील ठिपके (Septoria Leaf Spot)",
            "Spider mites Two-spotted spider mite": "दोन ठिपक्यांची कोळी (Spider Mites)",
            "Target Spot": "लक्ष्य ठिपके (Target Spot)",
            "Tomato Yellow Leaf Curl Virus": "पिवळा चुरडा-मुरडा विषाणू (TYLCV)",
            "healthy": "निरोगी पीक",
            "Healthy": "निरोगी पीक",
        }.get(disease, disease)

        # 1. Build Marathi Direct Answer
        if q_clean:
            if is_care:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} वरील '{disease_mr}' रोगाचा प्रतिबंध आणि काळजी घेण्यासाठी पुढील उपाययोजना करा: "
                    f"प्रमाणित रोगमुक्त बियाणे किंवा रोगप्रतिकारक वाणांची (उदा. बटाट्यासाठी कुफरी गिरिधारी, टोमॅटोसाठी अर्का रक्षक) निवड करा. "
                    f"पाऊस किंवा जास्त दमट हवामानात बीजाणूंचा प्रसार रोखण्यासाठी झाडांना मातीची योग्य भर (earthing-up) द्या. "
                    f"प्रतिबंधात्मक फवारणीसाठी मँकोझेब ७५% डब्ल्यूपी (२.५ ग्रॅम/लिटर) किंवा प्रोपिनेब ७०% डब्ल्यूपी (२.५ ग्रॅम/लिटर) पाण्यात मिसळून फवारा. "
                    f"प्रादुर्भाव वाढल्यास सायमोक्सानिल ८% + मँकोझेब ६४% डब्ल्यूपी (३.० ग्रॅम/लिटर) किंवा मेटलॅक्सिलयुक्त बुरशीनाशकाचा वापर करा."
                )
            elif is_where:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग प्रामुख्याने पिकाच्या जुन्या व खालच्या पानांवर प्रथम "
                    f"लहान गोलसर तपकिरी डागांच्या स्वरूपात सुरू होतो. रोग वाढल्यास तो खोडावर आणि शेवटी "
                    f"{'बटाट्याच्या कंदांवर' if 'बटाटा' in crop_mr else 'फळांवर'} पसरतो. "
                    f"हवेतील उष्ण व दमट वातावरणामुळे रोगाचा प्रसार जमिनीकडून वरील शेंड्याकडे होतो."
                )
            elif is_what:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} वरील '{disease_mr}' हा एक गंभीर बुरशीजन्य रोग आहे. "
                    f"यामुळे पानांवर गोलाकार वलयी (concentric rings) असलेले गडद तपकिरी डाग पडतात. "
                    f"पाने पिवळी पडून गळतात, ज्यामुळे प्रकाशसंश्लेषण कमी होऊन उत्पादनात मोठी घट होते."
                )
            elif is_why:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग बुरशीच्या बीजाणूंच्या (Fungal Spores) प्रादुर्भावामुळे होतो. "
                    f"सध्याचे तापमान ({weather.get('temperature_c', 25)}°C) आणि हवेतील आर्द्रता ({weather.get('humidity_percentage', 60)}%) "
                    f"या बीजाणूंची वाढ वेगाने होण्यासाठी अत्यंत अनुकूल ठरते."
                )
            elif is_symptom:
                symp = symptoms_points[0] if symptoms_points else "पानांवर गोल काळे-तपकिरी डाग व कडेने पिवळसर वलय दिसणे."
                direct_ans_mr = f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} {disease_mr} ची मुख्य लक्षणे: {symp}"
            elif is_neem:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): होय, सेंद्रिय नियंत्रणासाठी ५% निंबोळी अर्क (Neem Seed Extract) "
                    f"किंवा १०,००० ppm निंबोळी तेल २-३ मिली/लिटर पाण्यात मिसळून प्रतिबंधात्मक फवारणी करावी."
                )
            elif is_spray:
                best_spray = mgmt_points[0] if mgmt_points else "मँकोझेब ७५% WP (२.५ ग्रॅम/लिटर) किंवा कॉपर ऑक्सिक्लोराईड (२.५ ग्रॅम/लिटर)"
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): आयसीएआर (ICAR) शिफारशीनुसार, {crop_mr} वरील {disease_mr} च्या नियंत्रणासाठी "
                    f"मुख्य फवारणी: {best_spray}. फवारणी स्वच्छ सूर्यप्रकाश असताना किंवा सकाळी करावी."
                )
            elif is_water:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): पाणी देताना झाडांच्या पानांवर पाणी साचणार नाही याची काळजी घ्या. "
                    f"ठिबक सिंचनाचा वापर करा. हवेत आर्द्रता {weather.get('humidity_percentage', 60)}% असल्याने अतिपाणी देणे टाळा."
                )
            elif is_harvest:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): फवारणीनंतर रासायनिक औषधांचा प्रतीक्षा काळ (Pre-harvest interval) "
                    f"किमान ७ ते १० दिवस पाळावा. तोपर्यंत फळांची तोडणी करू नये."
                )
            else:
                ref_point = mgmt_points[0] if mgmt_points else "रोगट पाने त्वरित काढून टाका व तज्ज्ञांच्या सल्ल्याने फवारणी करा."
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} {disease_mr} बाबत आयसीएआर शिफारस अशी आहे: {ref_point} "
                    f"अधिक माहितीसाठी स्थानिक कृषी विज्ञान केंद्राशी संपर्क साधा."
                )
        else:
            direct_ans_mr = (
                f"तात्काळ उपाययोजना ({crop_mr} {disease_mr}): बाधित पाने गोळा करून शेताबाहेर नष्ट करा आणि "
                f"हवामानातील आर्द्रता लक्षात घेऊन शिफारशीत बुरशीनाशकाची फवारणी करा."
            )

        # 2. Build English Direct Answer
        if q_clean:
            if is_care:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): To effectively care for and prevent {crop} {disease}, follow these critical steps: "
                    f"Use certified disease-free seed tubers or resistant varieties (e.g. Kufri Girdhari for potato). Perform prophylactic earthing-up to prevent "
                    f"spores from washing down. Apply preventive sprays of Mancozeb 75% WP @ 2.5 g/L or Propineb 70% WP @ 2.5 g/L before symptoms spread, "
                    f"and ensure good field sanitation with proper crop rotation."
                )
            elif is_where:
                plant_parts = "tubers during harvest" if "potato" in crop.lower() else "fruits and stems"
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): This disease occurs primarily on the older, lower leaves "
                    f"close to the ground first, appearing as circular brown necrotic spots with target-board concentric rings. "
                    f"As the infection advances, it spreads upward through the foliage canopy, petioles, and eventually into the {plant_parts}. "
                    f"Splashing water and humid microclimates accelerate its vertical progression on the plant."
                )
            elif is_what:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): {crop} {disease} is a destructive fungal condition. "
                    f"It forms characteristic concentric rings on foliage, causing leaf chlorosis (yellowing), early defoliation, "
                    f"and significant yield loss if left unmanaged."
                )
            elif is_why:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): This occurs due to fungal spore infection favored by warm temperatures "
                    f"({weather.get('temperature_c', 25)}°C) and high ambient humidity ({weather.get('humidity_percentage', 60)}%). "
                    f"Spores overwinter in crop debris and spread via wind and rain splashes."
                )
            elif is_symptom:
                symp = symptoms_points[0] if symptoms_points else "Characteristic concentric dark-brown target rings on older foliage with yellow chlorotic halos."
                direct_ans_en = f"In response to your query ('{q_clean}'): Primary symptoms of {crop} {disease}: {symp}"
            elif is_neem:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Yes, 5% Neem Seed Kernel Extract (NSKE) or neem oil "
                    f"(10,000 ppm @ 2-3 ml/L) is effective as an eco-friendly preventive spray for {crop} against early disease progression. "
                    f"However, if foliar infection exceeds 10-15%, follow up with ICAR-recommended targeted fungicides."
                )
            elif is_spray:
                best_spray = mgmt_points[0] if mgmt_points else "Mancozeb 75% WP @ 2.0-2.5 g/L or Chlorothalonil 75% WP @ 2.0 g/L"
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): According to ICAR protocols, the primary chemical spray for "
                    f"{crop} {disease} is: {best_spray}. Ensure uniform spray coverage on both leaf surfaces."
                )
            elif is_water:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Avoid overhead sprinkler irrigation as wet foliage accelerates infection. "
                    f"Adopt drip irrigation and irrigate early in the morning so leaves dry quickly in sunlight."
                )
            elif is_harvest:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Strictly adhere to the recommended Pre-Harvest Interval (PHI) of 7-10 days "
                    f"after chemical fungicide application before picking fruits or tubers for consumption or market."
                )
            else:
                ref_point = mgmt_points[0] if mgmt_points else "Promptly rogue out infected foliage and consult ICAR spray schedules."
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Grounded advisory for {crop} {disease}: {ref_point}. "
                    f"Consult your nearest Krishi Vigyan Kendra (KVK) for localized guidance."
                )
        else:
            direct_ans_en = (
                f"Immediate Priority Action for {crop} ({disease}): Remove and destroy infected lower foliage immediately, "
                f"and apply protective fungicide spray considering prevailing weather conditions."
            )

        # Marathi guidance dictionary
        mr_guidance = {
            "direct_answer": direct_ans_mr,
            "explanation": f"हे पीक '{crop_mr}' असून मॉडेलने '{disease_mr}' चे निदान केले आहे (विश्वासार्हता: {confidence * 100:.1f}%). " + (f"लक्षणे: {symptoms_points[0]}" if symptoms_points else "पानांवर तपकिरी-काळे डाग पडणे हे प्रमुख लक्षण आहे."),
            "weather_interpretation": f"स्थानिक हवामान: तापमान {weather.get('temperature_c', 25)}°C, आर्द्रता {weather.get('humidity_percentage', 60)}%. {weather_note}",
            "management_guidance": "कृषी तज्ज्ञ शिफारशीनुसार उपाययोजना:\n" + ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else "• प्रादुर्भावग्रस्त पाने तोडून नष्ट करा.\n• ठिबक सिंचनाचा वापर करा आणि पानांवर पाणी साचू देऊ नका.\n• प्रमाणित बुरशीनाशकाचा योग्य प्रमाणात वापर करा."),
            "prevention": "दीर्घकालीन प्रतिबंधात्मक उपाय:\n• पिकांची योग्य फेरपालट (Crop Rotation) करा.\n• शेतात हवा खेळती राहण्यासाठी योग्य अंतर ठेवा.\n• प्रमाणित रोगमुक्त बियाणे किंवा वाणांचा वापर करा.",
            "precautions": "फवारणी करताना संरक्षक मास्क व हातमोजे वापरा. फवारणीनंतर तोडणीपूर्वीचा प्रतीक्षा काळ (PHI) पाळा.",
            "uncertainty_warning": f"सूचना: मॉडेलची विश्वासार्हता कमी ({confidence * 100:.1f}%) आहे. प्रत्यक्ष शेतात तज्ज्ञांकडून खात्री करा." if is_low_confidence else "",
            "expert_advisory": "मोठ्या प्रमाणात फवारणी करण्यापूर्वी आपल्या जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा तालुका कृषी अधिकाऱ्यांचा सल्ला घ्या.",
        }

        # English guidance dictionary
        en_guidance = {
            "direct_answer": direct_ans_en,
            "explanation": f"The leaf was classified as {crop} with '{disease}' by the MobileNetV3 vision model with {confidence * 100:.1f}% confidence. " + (f"Typical symptoms: {symptoms_points[0]}" if symptoms_points else "Necrotic foliar lesions are characteristic."),
            "weather_interpretation": f"Local weather context: Temperature: {weather.get('temperature_c', 25)}°C, Humidity: {weather.get('humidity_percentage', 60)}%. {weather_note}",
            "management_guidance": "Recommended Management (from Agricultural Knowledge Base):\n" + ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else "• Remove and safely dispose of severely infected foliage.\n• Ensure proper vine aeration and avoid overhead wetting.\n• Apply recommended protective fungicides or bio-agents at labelled rates."),
            "prevention": "Preventive Measures for Long-term Protection:\n• Practice 2-3 season crop rotation with non-host crops.\n• Maintain optimal plant spacing and stake indeterminate vines.\n• Use certified disease-free seeds and disease-resistant hybrids.",
            "precautions": "Wear PPE during chemical handling. Observe mandatory pre-harvest intervals.",
            "uncertainty_warning": f"Caution: Model confidence is low ({confidence * 100:.1f}%). Please confirm with agricultural authorities before treatment." if is_low_confidence else "",
            "expert_advisory": "Consult your local Krishi Vigyan Kendra (KVK) or district agricultural extension officer for on-field verification.",
        }

        res = dict(mr_guidance if is_marathi else en_guidance)
        res["en"] = en_guidance
        res["mr"] = mr_guidance
        res["direct_answer_en"] = direct_ans_en
        res["direct_answer_mr"] = direct_ans_mr
        res["language"] = "marathi" if is_marathi else "english"
        res["powered_by"] = "ICAR Agricultural Knowledge Base"
        return res


def get_gemini_service() -> GeminiGuidanceService:
    """Singleton getter for GeminiGuidanceService."""
    global _gemini_service
    if "_gemini_service" not in globals() or _gemini_service is None:
        _gemini_service = GeminiGuidanceService()
    return _gemini_service
