"""Gemini LLM Guidance Service for Evidence-Grounded Agricultural Advisory.

Integrates with the official Google GenAI SDK (`google-genai`), grounding LLM responses
on a tripartite synthesis combining THREE EQUAL INPUT PILLARS:
1. Crop Disease Detection (MobileNetV3 classification + ICAR RAG evidence)
2. Live Weather Context (Temperature, humidity, precipitation risks)
3. Soil Health Parameters (Synthetic soil pH, moisture %, N-P-K, organic carbon %)

Provides dual-language (English and Marathi) structured guidance with explicit
synthetic data disclosure and robust offline RAG fallback.
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
    """Service wrapper for Google Gemini with multi-model fallback and tripartite grounding."""

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
        soil: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate structured agricultural guidance with multi-model fallback and tripartite EN/MR outputs."""
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
                        soil=soil,
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
                            soil=soil,
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
            soil=soil,
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
        soil: dict[str, Any] | None = None,
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
            soil=soil,
        )

        en_block = guidance.get("en")
        if not isinstance(en_block, dict) or not en_block.get("explanation"):
            if not is_marathi and guidance.get("explanation"):
                en_block = {
                    "direct_answer": guidance.get("direct_answer", ""),
                    "explanation": guidance.get("explanation", ""),
                    "weather_interpretation": guidance.get("weather_interpretation", ""),
                    "soil_interpretation": guidance.get("soil_interpretation", fallback.get("en", {}).get("soil_interpretation", "")),
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
                    "soil_interpretation": guidance.get("soil_interpretation", fallback.get("mr", {}).get("soil_interpretation", "")),
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
        if "soil_interpretation" not in guidance:
            guidance["soil_interpretation"] = mr_block.get("soil_interpretation") if is_marathi else en_block.get("soil_interpretation", "")

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
        soil: dict[str, Any] | None = None,
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
            f"Microclimate Risk Note: {weather.get('risk_analysis', 'N/A')}"
        )

        if soil:
            interp = soil.get("interpretation", {})
            soil_summary = (
                f"Location / Agro-climatic Zone: {soil.get('location', 'Maharashtra')}, "
                f"Soil Type: {soil.get('soil_type', 'Medium Black Vertisol')}, "
                f"pH: {soil.get('soil_ph', 'N/A')} ({interp.get('ph_status', 'Neutral')}), "
                f"Moisture: {soil.get('soil_moisture_percent', 'N/A')}% ({interp.get('moisture_status', 'Optimal')}), "
                f"Available Nitrogen (N): {soil.get('nitrogen_kg_ha', 'N/A')} kg/ha ({interp.get('nitrogen_rating', 'Medium')}), "
                f"Available Phosphorus (P): {soil.get('phosphorus_kg_ha', 'N/A')} kg/ha ({interp.get('phosphorus_rating', 'Medium')}), "
                f"Available Potassium (K): {soil.get('potassium_kg_ha', 'N/A')} kg/ha ({interp.get('potassium_rating', 'High')}), "
                f"Organic Carbon: {soil.get('organic_carbon_percent', 'N/A')}%, "
                f"Electrical Conductivity: {soil.get('electrical_conductivity_ds_m', 'N/A')} dS/m. "
                f"[TAG: SYNTHETIC DEMONSTRATION DATA (Record ID: {soil.get('record_id', 'SYN-SOIL')})]. "
                f"Inter-pillar Risk: {interp.get('disease_soil_interaction', 'Normal regional baseline')}."
            )
        else:
            soil_summary = "Soil Health: [DATA NOT RECORDED / UNAVAILABLE]. (Proceed with standard foliar agronomic principles)."

        has_specific_question = bool(question and question.strip())

        if has_specific_question:
            qa_instruction = f"""CRITICAL Q&A INSTRUCTION:
The farmer asked this SPECIFIC QUESTION: "{question.strip()}"
You MUST provide a direct, comprehensive, empathetic, and highly actionable answer addressing this EXACT question in the "direct_answer" field of BOTH the "en" and "mr" blocks.
- Synthesize all three input pillars: explain how the combination of classified leaf disease, prevailing weather, and soil health parameters answers the farmer's question.
- If the farmer asks "how to prevent", "what care should be taken", or "how to control", specify exact varieties, prophylactic earthing up, sanitation, irrigation scheduling adjusted for soil moisture, and chemical/organic sprays with dosage (e.g. Mancozeb 75% WP @ 2.5 g/L).
- If the farmer asks about soil nutrients, explain how soil pH or potassium levels influence disease tolerance without making unsupported synthetic fertilizer prescriptions.
- Make the "direct_answer" the centerpiece of your advisory in BOTH English ("en") and authentic Marathi ("mr")."""
        else:
            qa_instruction = """No specific question was submitted. In the "direct_answer" field of both "en" and "mr", provide a clear 2-3 sentence executive summary of the most urgent immediate next step the farmer must take today, synthesizing crop symptoms, weather, and soil status."""

        is_general_inquiry = "general" in (crop or "").lower() or "general" in (disease or "").lower()

        prompt = f"""You are an expert AI Agriculture Assistant advising an Indian farmer in Maharashtra.
Your role is to provide safe, actionable, and grounded agricultural guidance by synthesizing THREE EQUALLY IMPORTANT INPUT CATEGORIES:
1. CROP DISEASE DETECTION: MobileNetV3 visual classification and ICAR verified knowledge base.
2. WEATHER MICROCLIMATE: Live regional temperature, ambient humidity, and rainfall conditions.
3. SOIL HEALTH PARAMETERS: Synthetic soil pH, moisture %, N-P-K, and organic carbon baseline.

CRITICAL CONSTRAINTS:
1. {'Do NOT claim you diagnosed the leaf image. The disease classification was performed separately by a MobileNetV3 computer vision model.' if not is_general_inquiry else 'You are advising on a general crop management query without an image. Ground your advice in ICAR agronomic principles.'}
2. Ground all biological, cultural, and chemical advice strictly in the provided Agricultural Evidence below. Do NOT invent unsupported pesticide dosages or dangerous chemical mixtures.
3. EQUAL THREE-PILLAR SYNTHESIS: Cross-correlate all 3 pillars in your advisory:
   - How ambient weather (humidity/temperature) combined with soil moisture creates microclimates favoring or inhibiting pathogen reproduction.
   - How soil nutrients (e.g. potassium for cell wall strength vs excess nitrogen leading to succulent tissue) impact foliar disease susceptibility.
   - Adjust irrigation timing considering current soil moisture and weather.
4. SYNTHETIC SOIL DATA DISCLOSURE: Explicitly state in the advisory that the soil values are synthetic baseline demonstration data for academic illustration, and recommend an official soil testing laboratory (Soil Health Card) for physical field fertilizer scheduling.
5. Output MUST be valid JSON containing TWO top-level objects: "en" (English guidance) and "mr" (Marathi मार्गदर्शन).
{'6. The disease prediction has LOW CONFIDENCE. Emphasize uncertainty and advise the farmer to consult their local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying chemicals.' if is_low_confidence else '6. Always remind the farmer to confirm with local agricultural officers before large-scale chemical applications.'}

{qa_instruction}

INPUT DATA (3 EQUAL PILLARS):
- [PILLAR 1 - DISEASE CONTEXT]: Crop: {crop or 'General Solanaceae'}, Disease: {disease or 'Agronomic Inquiry'}, Model Confidence: {confidence * 100:.1f}% ({'LOW CONFIDENCE' if is_low_confidence else 'NORMAL'})
- [PILLAR 2 - WEATHER CONTEXT]: {weather_summary}
- [PILLAR 3 - SOIL CONTEXT]: {soil_summary}
- Farmer's Question: {question or 'What should I do to treat and protect my crop?'}

RETRIEVED AGRICULTURAL EVIDENCE (ICAR / TNAU):
{evidence_text if evidence_text else 'Standard ICAR Horticultural Practices'}

Please return your response in JSON format with exactly TWO top-level objects: "en" and "mr":
{{
  "en": {{
    "direct_answer": "Comprehensive, empathetic, and actionable answer in English synthesizing disease, weather, and soil.",
    "explanation": "Simple, clear explanation in English of what this condition is, why it occurred, and how weather + soil contributed.",
    "weather_interpretation": "How current temperature and humidity impact this disease or spray timing in English.",
    "soil_interpretation": "Detailed assessment of soil pH, moisture %, and N-P-K impact on crop disease resistance in English, with synthetic data disclosure.",
    "management_guidance": "Clear bulleted cultural, organic, and chemical actions with dosage in English.",
    "prevention": "Practical measures in English to protect next crop cycle or healthy foliage.",
    "precautions": "Safety measures, PPE, and pre-harvest interval in English.",
    "expert_advisory": "Recommendation in English to consult local KVK or extension officer and test physical soil."
  }},
  "mr": {{
    "direct_answer": "शेतकऱ्याच्या प्रश्नाचे थेट, आदरयुक्त आणि रोग-हवामान-मृदा या तिन्ही घटकांचा समन्वय साधणारे परिपूर्ण मराठीत उत्तर.",
    "explanation": "हा रोग काय आहे, कशामुळे झाला आणि हवामान व जमिनीतील ओलाव्याचा त्यावर कसा परिणाम झाला याचे सोप्या मराठीत स्पष्टीकरण.",
    "weather_interpretation": "सध्याचे तापमान आणि हवेतील आर्द्रता या रोगावर कसा परिणाम करते याचे मराठीत विश्लेषण.",
    "soil_interpretation": "जमिनीचा सामू (pH), ओलावा % आणि नत्र-स्फुरद-पालाश यांचा पिकाच्या रोगावरील परिणामाचे मराठीत विश्लेषण (प्रात्यक्षिक माहिती निर्देशनासह).",
    "management_guidance": "आयसीएआर शिफारशीनुसार उपाययोजना, सेंद्रिय व रासायनिक फवारणीचे प्रमाण मराठीत.",
    "prevention": "पुढील हंगामासाठी किंवा निरोगी झाडांसाठी दीर्घकालीन प्रतिबंधात्मक उपाय मराठीत.",
    "precautions": "फवारणी करताना घ्यावयाची काळजी आणि तोडणीपूर्वीचा प्रतीक्षा काळ मराठीत.",
    "expert_advisory": "जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा कृषी अधिकाऱ्यांचा सल्ला आणि प्रत्यक्ष माती परीक्षण करण्याची शिफारस."
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
        soil: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        prompt = self._build_prompt(
            crop, disease, confidence, weather, evidence, question, is_marathi, is_low_confidence, soil=soil
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
        soil: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evidence-grounded fallback synthesized directly from RAG documents with tripartite intent matching."""
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

        # Soil parameters synthesis
        soil_ph = soil.get("soil_ph", 7.0) if soil else 7.0
        soil_moisture = soil.get("soil_moisture_percent", 25.0) if soil else 25.0
        soil_type = soil.get("soil_type", "Medium Black Soil (Vertisol)") if soil else "Medium Black Soil"
        soil_k = soil.get("potassium_kg_ha", 300.0) if soil else 300.0
        soil_n = soil.get("nitrogen_kg_ha", 185.0) if soil else 185.0
        soil_interp = soil.get("interpretation", {}) if soil else {}
        soil_type_mr = soil_interp.get("soil_type_mr", "मध्यम काळी कसदार जमीन")
        dis_soil_note_en = soil_interp.get("disease_soil_interaction", "Soil moisture and nutrient levels are at balanced regional baseline.")
        dis_soil_note_mr = soil_interp.get("disease_soil_interaction_mr", "जमिनीतील गुणधर्म संतुलित असून पीक पोषणासाठी अनुकूल आहेत.")

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
        is_soil = any(k in q_lower for k in ["soil", "fertilizer", "khad", "npk", "ph", "carbon", "माती", "मृदा", "खत", "खते"])

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

        # 1. Build Marathi Direct Answer (Synthesizing Disease + Weather + Soil)
        if q_clean:
            if is_care:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} वरील '{disease_mr}' रोगाचा प्रतिबंध आणि काळजी घेण्यासाठी पीक-हवामान-माती या तिन्ही घटकांचा विचार करून पुढील उपाय करा: "
                    f"१) प्रमाणित रोगमुक्त बियाणे किंवा रोगप्रतिकारक वाण (उदा. कुफरी गिरिधारी, अर्का रक्षक) वापरा. "
                    f"२) जमिनीत ओलावा {soil_moisture:.1f}% असल्याने अतिपाणी देणे टाळा आणि पाण्याचा निचरा व्यवस्थित ठेवा; बीजाणू कंदांपर्यंत न पोहोचण्यासाठी झाडांना मातीची भर (earthing-up) द्या. "
                    f"३) हवेतील दमट वातावरणात प्रतिबंधात्मक फवारणीसाठी मँकोझेब ७५% डब्ल्यूपी (२.५ ग्रॅम/लिटर) पाण्यात मिसळून फवारा. "
                    f"प्रादुर्भाव वाढल्यास सायमोक्सानिल ८% + मँकोझेब ६४% डब्ल्यूपी (३.० ग्रॅम/लिटर) चा वापर करा. (टीप: मातीची माहिती प्रात्यक्षिक अभ्यासासाठी आहे)."
                )
            elif is_soil:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): उपलब्ध प्रात्यक्षिक मृदा विश्लेषणानुसार, जमिनीचा सामू {soil_ph} असून तो {crop_mr} पिकासाठी अनुकूल आहे. "
                    f"जमिनीत पालाशचे प्रमाण {soil_k:.0f} किलो/हेक्टर (चांगले) असल्याने पानांमधील पेशींना बळकटी मिळते. "
                    f"नत्र {soil_n:.0f} किलो/हेक्टर असल्याने नत्राचा अतिवापर टाळावा, अन्यथा पिकाची कोवळी पाने रोगास बळी पडतात. "
                    f"(सूचना: ही प्रात्यक्षिक नोंद असून खतांच्या निश्चित प्रमाणासाठी अधिकृत माती परीक्षण प्रयोगशाळेचा सल्ला घ्यावा)."
                )
            elif is_where:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग प्रामुख्याने पिकाच्या जुन्या व खालच्या पानांवर प्रथम "
                    f"लहान गोलसर तपकिरी डागांच्या स्वरूपात सुरू होतो. जमिनीत ओलावा {soil_moisture:.1f}% असल्यास जमिनीकडील पानांवर रोग वेगाने वाढतो. "
                    f"रोग वाढल्यास तो खोडावर आणि शेवटी {'बटाट्याच्या कंदांवर' if 'बटाटा' in crop_mr else 'फळांवर'} पसरतो."
                )
            elif is_what:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} वरील '{disease_mr}' हा एक गंभीर बुरशीजन्य रोग आहे. "
                    f"यामुळे पानांवर गोलाकार वलयी (concentric rings) असलेले गडद तपकिरी डाग पडतात. "
                    f"सध्याचे तापमान ({weather.get('temperature_c', 25)}°C) व मातीतील ओलावा ({soil_moisture:.1f}%) यामुळे या रोगाचा प्रादुर्भाव वाढू शकतो."
                )
            elif is_why:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग बुरशीच्या बीजाणूंच्या प्रादुर्भावामुळे होतो. "
                    f"सध्याचे तापमान ({weather.get('temperature_c', 25)}°C), हवेतील आर्द्रता ({weather.get('humidity_percentage', 60)}%) "
                    f"आणि मातीतील ओलावा ({soil_moisture:.1f}%) हे बीजाणूंची वाढ वेगाने होण्यासाठी अनुकूल ठरतात."
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
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): मातीतील ओलावा {soil_moisture:.1f}% असल्याने झाडांच्या पानांवर पाणी साचणार नाही याची काळजी घ्या. "
                    f"ठिबक सिंचनाचा वापर करा आणि तुषार सिंचन टाळा. पाण्याचा निचरा चांगला ठेवा."
                )
            elif is_harvest:
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): फवारणीनंतर रासायनिक औषधांचा प्रतीक्षा काळ (Pre-harvest interval) "
                    f"किमान ७ ते १० दिवस पाळावा. तोपर्यंत फळांची तोडणी करू नये."
                )
            else:
                ref_point = mgmt_points[0] if mgmt_points else "रोगट पाने त्वरित काढून टाका व तज्ज्ञांच्या सल्ल्याने फवारणी करा."
                direct_ans_mr = (
                    f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop_mr} {disease_mr} बाबत कृषी पुरावे: {ref_point} "
                    f"मातीचा सामू {soil_ph} व ओलावा {soil_moisture:.1f}% आहे. अधिक मार्गदर्शनासाठी कृषी विज्ञान केंद्राशी संपर्क साधा."
                )
        else:
            direct_ans_mr = (
                f"तात्काळ एकात्मिक उपाययोजना ({crop_mr} {disease_mr}): बाधित पाने गोळा करून शेताबाहेर नष्ट करा. "
                f"हवामानातील आर्द्रता ({weather.get('humidity_percentage', 60)}%) व जमिनीतील ओलावा ({soil_moisture:.1f}%) लक्षात घेऊन "
                f"पाण्याचा निचरा सुरळीत ठेवा आणि शिफारशीत बुरशीनाशकाची फवारणी करा."
            )

        # 2. Build English Direct Answer (Synthesizing Disease + Weather + Soil)
        if q_clean:
            if is_care:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): To care for and prevent {crop} {disease}, follow a tripartite crop-weather-soil strategy: "
                    f"1) Use certified disease-free seed tubers or resistant varieties (e.g. Kufri Girdhari for potato). "
                    f"2) With soil moisture at {soil_moisture:.1f}% and prevailing humidity ({weather.get('humidity_percentage', 60)}%), avoid overhead irrigation and perform prophylactic earthing-up to prevent zoospores from washing down into the root zone. "
                    f"3) Apply protective foliar sprays of Mancozeb 75% WP @ 2.5 g/L or Propineb 70% WP @ 2.5 g/L before foliar infection spreads. (Note: soil values reflect synthetic demonstration baselines)."
                )
            elif is_soil:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Based on regional synthetic soil benchmarking, the soil pH is {soil_ph} ({soil_interp.get('ph_status', 'Neutral')}) and available potassium is {soil_k:.0f} kg/ha (adequate). "
                    f"Adequate potassium strengthens foliar cell walls against pathogen entry. Available nitrogen is {soil_n:.0f} kg/ha; avoid excess top-dressed nitrogen as succulent tissue is prone to blight. "
                    f"(Notice: This is synthetic demonstration data; consult an authorized soil testing laboratory for physical field fertilizer cards)."
                )
            elif is_where:
                plant_parts = "tubers during harvest" if "potato" in crop.lower() else "fruits and stems"
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): This disease occurs primarily on the older, lower leaves "
                    f"near the soil first. Soil moisture ({soil_moisture:.1f}%) and humid microclimates facilitate fungal sporulation close to the ground, "
                    f"eventually progressing upward into the canopy and {plant_parts}."
                )
            elif is_what:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): {crop} {disease} is a destructive foliar disease. "
                    f"It forms necrotic concentric lesions on foliage, causing early defoliation. "
                    f"Current temperature ({weather.get('temperature_c', 25)}°C) and soil moisture ({soil_moisture:.1f}%) create active conditions for spread."
                )
            elif is_why:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): This condition develops when fungal spores encounter warm temperatures "
                    f"({weather.get('temperature_c', 25)}°C), elevated humidity ({weather.get('humidity_percentage', 60)}%), and moist soil ({soil_moisture:.1f}%). "
                    f"Pathogens overwinter in crop residues and spread via wind and rain splashes."
                )
            elif is_symptom:
                symp = symptoms_points[0] if symptoms_points else "Concentric dark-brown target rings on older foliage with yellow chlorotic halos."
                direct_ans_en = f"In response to your query ('{q_clean}'): Primary symptoms of {crop} {disease}: {symp}"
            elif is_neem:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Yes, 5% Neem Seed Kernel Extract (NSKE) or neem oil "
                    f"(10,000 ppm @ 2-3 ml/L) is effective as an eco-friendly preventive foliar spray against early progression. "
                    f"However, if foliar blight exceeds 10%, apply ICAR-recommended registered fungicides."
                )
            elif is_spray:
                best_spray = mgmt_points[0] if mgmt_points else "Mancozeb 75% WP @ 2.0-2.5 g/L or Chlorothalonil 75% WP @ 2.0 g/L"
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): According to ICAR protocols, the primary chemical spray for "
                    f"{crop} {disease} is: {best_spray}. Spray early morning when leaves are dry."
                )
            elif is_water:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): With current soil moisture at {soil_moisture:.1f}%, adopt drip irrigation "
                    f"and avoid overhead sprinkling. Ensure excellent bed drainage so roots are not waterlogged."
                )
            elif is_harvest:
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Adhere strictly to the recommended Pre-Harvest Interval (PHI) of 7-10 days "
                    f"after fungicide application before harvesting produce for market or consumption."
                )
            else:
                ref_point = mgmt_points[0] if mgmt_points else "Promptly rogue out infected foliage and consult ICAR spray schedules."
                direct_ans_en = (
                    f"In response to your query ('{q_clean}'): Tripartite advisory for {crop} {disease}: {ref_point}. "
                    f"Soil pH is {soil_ph} with {soil_moisture:.1f}% moisture. Consult your local Krishi Vigyan Kendra (KVK)."
                )
        else:
            direct_ans_en = (
                f"Immediate Integrated Action for {crop} ({disease}): Remove and destroy lower infected foliage immediately. "
                f"Given {weather.get('humidity_percentage', 60)}% ambient humidity and {soil_moisture:.1f}% soil moisture, regulate irrigation "
                f"and apply protective fungicide spray according to ICAR schedules."
            )

        # Marathi guidance dictionary
        mr_guidance = {
            "direct_answer": direct_ans_mr,
            "explanation": f"हे पीक '{crop_mr}' असून मॉडेलने '{disease_mr}' चे निदान केले आहे (विश्वासार्हता: {confidence * 100:.1f}%). " + (f"लक्षणे: {symptoms_points[0]}" if symptoms_points else "पानांवर तपकिरी-काळे डाग पडणे हे प्रमुख लक्षण आहे."),
            "weather_interpretation": f"स्थानिक हवामान: तापमान {weather.get('temperature_c', 25)}°C, आर्द्रता {weather.get('humidity_percentage', 60)}%. {weather_note}",
            "soil_interpretation": (
                f"मृदा आरोग्य विश्लेषण [प्रात्यक्षिक नोंद]: जमिनीचा प्रकार '{soil_type_mr}' असून सामू (pH) {soil_ph} ({soil_interp.get('ph_status_mr', 'योग्य')}) आणि ओलावा {soil_moisture:.1f}% ({soil_interp.get('moisture_status_mr', 'वाफसा')}) आहे. "
                f"उपलब्ध नत्र: {soil_n:.0f} kg/ha ({soil_interp.get('nitrogen_rating_mr', 'मध्यम')}), पालाश: {soil_k:.0f} kg/ha ({soil_interp.get('potassium_rating_mr', 'मुबलक')}). {dis_soil_note_mr} "
                f"(टीप: ही प्रात्यक्षिक माहिती आहे; प्रत्यक्ष खत व्यवस्थापनासाठी अधिकृत माती परीक्षण पत्रिका वापरावी)."
            ),
            "management_guidance": "एकात्मिक कृषी तज्ज्ञ शिफारशीनुसार उपाययोजना:\n" + ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else "• प्रादुर्भावग्रस्त पाने तोडून नष्ट करा.\n• ठिबक सिंचनाचा वापर करा आणि पानांवर पाणी साचू देऊ नका.\n• प्रमाणित बुरशीनाशकाचा योग्य प्रमाणात वापर करा."),
            "prevention": "दीर्घकालीन प्रतिबंधात्मक उपाय:\n• पिकांची योग्य फेरपालट (Crop Rotation) करा.\n• शेतात हवा खेळती राहण्यासाठी योग्य अंतर ठेवा.\n• प्रमाणित रोगमुक्त बियाणे किंवा वाणांचा वापर करा.\n• जमिनीतील सेंद्रिय कर्ब वाढवण्यासाठी सेंद्रिय खतांचा वापर करा.",
            "precautions": "फवारणी करताना संरक्षक मास्क व हातमोजे वापरा. फवारणीनंतर तोडणीपूर्वीचा प्रतीक्षा काळ (PHI: ७-१० दिवस) पाळा.",
            "uncertainty_warning": f"सूचना: मॉडेलची विश्वासार्हता कमी ({confidence * 100:.1f}%) आहे. प्रत्यक्ष शेतात तज्ज्ञांकडून खात्री करा." if is_low_confidence else "",
            "expert_advisory": "मोठ्या प्रमाणात फवारणी करण्यापूर्वी आपल्या जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा तालुका कृषी अधिकाऱ्यांचा सल्ला घ्या.",
        }

        # English guidance dictionary
        en_guidance = {
            "direct_answer": direct_ans_en,
            "explanation": f"The leaf was classified as {crop} with '{disease}' by the MobileNetV3 vision model with {confidence * 100:.1f}% confidence. " + (f"Typical symptoms: {symptoms_points[0]}" if symptoms_points else "Necrotic foliar lesions are characteristic."),
            "weather_interpretation": f"Local weather context: Temperature: {weather.get('temperature_c', 25)}°C, Humidity: {weather.get('humidity_percentage', 60)}%. {weather_note}",
            "soil_interpretation": (
                f"Soil Health Context [Synthetic Demonstration Data]: Soil type is '{soil_type}', pH is {soil_ph} ({soil_interp.get('ph_status', 'Optimal')}), and moisture is {soil_moisture:.1f}% ({soil_interp.get('moisture_status', 'Field Capacity')}). "
                f"Available N: {soil_n:.0f} kg/ha ({soil_interp.get('nitrogen_rating', 'Medium')}), K: {soil_k:.0f} kg/ha ({soil_interp.get('potassium_rating', 'High')}). {dis_soil_note_en} "
                f"(Disclaimer: Synthetic demonstration record. Consult a certified Soil Health testing laboratory for field fertilizer prescription)."
            ),
            "management_guidance": "Recommended Integrated Management (from Agricultural Knowledge Base):\n" + ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else "• Remove and safely dispose of severely infected foliage.\n• Ensure proper vine aeration and avoid overhead wetting.\n• Apply recommended protective fungicides or bio-agents at labelled rates."),
            "prevention": "Preventive Measures for Long-term Protection:\n• Practice 2-3 season crop rotation with non-host crops.\n• Maintain optimal plant spacing and stake indeterminate vines.\n• Use certified disease-free seeds and disease-resistant hybrids.\n• Enhance soil organic matter with well-decomposed farmyard manure.",
            "precautions": "Wear PPE during chemical handling. Observe mandatory pre-harvest intervals.",
            "uncertainty_warning": f"Caution: Model confidence is low ({confidence * 100:.1f}%). Please confirm with agricultural authorities before treatment." if is_low_confidence else "",
            "expert_advisory": "Consult your local Krishi Vigyan Kendra (KVK) or district agricultural extension officer for on-field verification.",
        }

        res = dict(mr_guidance if is_marathi else en_guidance)
        res["en"] = en_guidance
        res["mr"] = mr_guidance
        res["direct_answer_en"] = direct_ans_en
        res["direct_answer_mr"] = direct_ans_mr
        res["soil_interpretation"] = mr_guidance["soil_interpretation"] if is_marathi else en_guidance["soil_interpretation"]
        res["language"] = "marathi" if is_marathi else "english"
        res["powered_by"] = "ICAR Agricultural Knowledge Base"
        return res


# Singleton getter
_gemini_service = None


def get_gemini_service() -> GeminiGuidanceService:
    """Singleton getter for GeminiGuidanceService."""
    global _gemini_service
    if "_gemini_service" not in globals() or _gemini_service is None:
        _gemini_service = GeminiGuidanceService()
    return _gemini_service
