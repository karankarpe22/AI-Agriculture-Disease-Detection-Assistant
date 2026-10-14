"""Phase 10 & 13: Google Gemini Generative AI Guidance Layer with Multi-Model Fallback & Contextual Q&A."""
from __future__ import annotations

import json
import os
import re
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# Priority models: uses high-capacity, low-latency models first to avoid 429 Quota Exhaustion
DEFAULT_MODELS = [
    os.getenv("GEMINI_MODEL", "").strip() or "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
]
# Filter out empty entries
DEFAULT_MODELS = [m for m in DEFAULT_MODELS if m]


class GeminiGuidanceService:
    """Provides evidence-grounded, contextual agricultural guidance using Google Gemini."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "").strip() or "gemini-flash-lite-latest"
        self._client = None

    def _get_client(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Google GenAI client: {e}")
                return None
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
        """Generate structured agricultural guidance with multi-model fallback."""
        lang = (language or "english").lower().strip()
        is_marathi = lang in ["marathi", "mr", "मराठी"]

        # If Gemini API key is available, call official Google GenAI SDK across candidate models
        client = self._get_client()
        if client:
            # Build list of candidate models starting with preferred model
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
                        return guidance
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                    is_busy = "503" in err_str or "UNAVAILABLE" in err_str
                    if is_rate_limit or is_busy:
                        print(f"Gemini model '{m}' rate-limited or busy ({'429 Quota' if is_rate_limit else '503 Busy'}). Trying next candidate model...")
                        continue
                    else:
                        print(f"Gemini call on '{m}' failed ({e}). Trying next model...")
                        continue

            print("All Gemini candidate models were exhausted or unavailable. Falling back to grounded RAG formatter.")

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
            f"Risk note: {weather.get('risk_analysis', 'N/A')}"
        )

        # Build Soil Context Summary
        soil_summary = "Soil condition: Regional baseline (Live sensor metrics unavailable)."
        if soil and isinstance(soil, dict):
            params = soil.get("soil_parameters", {})
            if not isinstance(params, dict):
                params = {}
            st = soil.get("soil_type", {})
            if isinstance(st, dict):
                st_name = st.get("name") or st.get("order") or "Regional agricultural soil"
                st_text = st.get("texture", "")
            elif isinstance(st, str):
                st_name = st
                st_text = ""
            else:
                st_name = "Regional agricultural soil"
                st_text = ""
            
            rz = params.get("soil_moisture_root_zone", {}) if isinstance(params.get("soil_moisture_root_zone"), dict) else {}
            m_val = rz.get("value")
            m_unit = rz.get("unit", "m³/m³")
            m_interp = rz.get("interpretation", "")
            
            stemp = params.get("soil_temperature_surface", {}) if isinstance(params.get("soil_temperature_surface"), dict) else {}
            t_val = stemp.get("value")
            
            sph = params.get("soil_ph", {}) if isinstance(params.get("soil_ph"), dict) else {}
            ph_val = sph.get("value")
            ph_interp = sph.get("interpretation", "")
            
            loc = soil.get("location", {}) if isinstance(soil.get("location"), dict) else {}
            resolved_loc = loc.get("resolved_name") or loc.get("district") or ""
            
            soil_parts = [f"Location: {resolved_loc}" if resolved_loc else ""]
            soil_parts.append(f"Soil Type/Order: {st_name} ({st_text})" if st_text else f"Soil Type/Order: {st_name}")
            if m_val is not None:
                soil_parts.append(f"Root-Zone Moisture (3-9cm): {m_val} {m_unit} [{m_interp or 'Live land surface model'}]")
            if t_val is not None:
                soil_parts.append(f"Soil Surface Temp: {t_val}°C")
            if ph_val:
                soil_parts.append(f"Typical Regional pH: {ph_val} [{ph_interp or 'Survey baseline'}]")
            soil_parts.append("Note: Available N-P-K (kg/ha) require physical laboratory testing and are unavailable via satellite/remote queries.")
            soil_summary = "; ".join([p for p in soil_parts if p])

        target_lang = "Marathi (मराठी)" if is_marathi else "English"
        has_specific_question = bool(question and question.strip())

        if has_specific_question:
            qa_instruction = f"""CRITICAL Q&A INSTRUCTION:
The farmer asked this SPECIFIC QUESTION: "{question.strip()}"
You MUST provide a direct, comprehensive, empathetic, and highly actionable answer addressing this EXACT question in the "direct_answer" field.
- If the farmer asks "where does this occur", explain precisely which plant organs (lower foliage, stems, tubers, flowers, fruits), canopy position, and field conditions it appears on.
- If the farmer asks "what is this" or "what", define the disease, the causative fungal/bacterial pathogen, and what it does to the plant.
- If the farmer asks about a specific chemical, organic remedy (like neem oil), watering schedule, or harvest safety, address it directly with exact dosage/timing grounded in ICAR evidence.
- Do NOT give a vague reply. Make the "direct_answer" the centerpiece of your advisory in {target_lang}."""
        else:
            qa_instruction = """No specific question was submitted. In the "direct_answer" field, provide a clear 2-3 sentence executive summary of the most urgent immediate next step the farmer must take today to protect their crop."""

        is_general_inquiry = "general" in (crop or "").lower() or "general" in (disease or "").lower()

        prompt = f"""You are an expert AI Agriculture Assistant advising an Indian farmer.
Your role is to provide safe, actionable, and grounded agricultural guidance.

CRITICAL CONSTRAINTS:
1. {'Do NOT claim you diagnosed the leaf image. The disease classification was performed separately by a MobileNetV3 computer vision model.' if not is_general_inquiry else 'You are advising on a general crop management query without an image. Ground your advice in ICAR agronomic principles.'}
2. Ground all biological, cultural, and chemical advice strictly in the provided Agricultural Evidence below. Do NOT invent unsupported pesticide dosages or dangerous chemical mixtures.
3. Consider the provided Weather Context when discussing disease spread or spray timing.
4. Response language MUST be {target_lang}.
{'5. The disease prediction has LOW CONFIDENCE. Emphasize uncertainty and advise the farmer to consult their local Krishi Vigyan Kendra (KVK) or agriculture officer before spraying chemicals.' if is_low_confidence else '5. Always remind the farmer to confirm with local agricultural officers before large-scale chemical applications.'}

{qa_instruction}

INPUT DATA:
- Classified Crop: {crop or 'General Solanaceae / Agriculture'}
- Classified Disease / Topic: {disease or 'Agronomic Inquiry'}
- Model Confidence: {confidence * 100:.1f}% ({'LOW CONFIDENCE' if is_low_confidence else 'NORMAL'})
- Weather Context: {weather_summary}
- Live & Regional Soil Health Context: {soil_summary}
- Farmer's Question: {question or 'What should I do to treat and protect my crop?'}

RETRIEVED AGRICULTURAL EVIDENCE:
{evidence_text if evidence_text else 'Standard ICAR Horticultural Practices'}

Please return your response in JSON format with exactly these keys:
{{
  "direct_answer": "Direct, empathetic, and comprehensive answer to the farmer's specific question (or immediate executive recommendation if no question was asked).",
  "explanation": "A simple, clear explanation of what this condition is and why it occurred, integrating weather and soil realities.",
  "weather_interpretation": "How current atmospheric temperature and humidity impact this disease or spray timing.",
  "soil_interpretation": "Agronomic analysis of how current soil moisture, soil temperature, and soil type affect root health, waterlogging risk, and pathogen proliferation.",
  "management_guidance": "Clear, bulleted step-by-step cultural, organic, and recommended chemical actions.",
  "prevention": "Practical measures to protect the next crop cycle or unaffected plants.",
  "precautions": "Safety measures, pre-harvest interval, or spray cautions.",
  "uncertainty_warning": "Warning about prediction confidence and caution if confidence is low, otherwise empty or short reassurance.",
  "expert_advisory": "Recommendation to consult local agricultural officers/KVK."
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
        # Clean potential markdown wrapping
        clean_text = re.sub(r"^```json\s*", "", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"^```\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text).strip()

        try:
            parsed = json.loads(clean_text)
            parsed["language"] = "marathi" if is_marathi else "english"
            parsed["powered_by"] = f"Google Gemini ({model})"

            # Ensure direct_answer exists in parsed result
            if "direct_answer" not in parsed or not parsed["direct_answer"]:
                if question:
                    parsed["direct_answer"] = (
                        f"तुमच्या प्रश्नासाठी ({question}): कृपया खालील उपाययोजना आणि मार्गदर्शनाचा अवलंब करा."
                        if is_marathi
                        else f"In response to your query ('{question}'): Please follow the ICAR-recommended management practices detailed below."
                    )
                else:
                    parsed["direct_answer"] = (
                        "पिकाचे रक्षण करण्यासाठी तात्काळ बाधित पाने गोळा करून नष्ट करा आणि शिफारशीनुसार फवारणी करा."
                        if is_marathi
                        else f"Immediate recommended action for {crop}: Remove affected foliage and apply recommended protective spray."
                    )

            return parsed
        except Exception:
            fallback_answer = clean_text[:400]
            return {
                "direct_answer": fallback_answer,
                "explanation": clean_text,
                "weather_interpretation": weather.get("risk_analysis", ""),
                "management_guidance": "Please refer to the evidence sources cited below.",
                "prevention": "Implement crop rotation and balanced irrigation.",
                "precautions": "Wear personal protective equipment during any spray.",
                "uncertainty_warning": "Low prediction confidence" if is_low_confidence else "",
                "expert_advisory": "Consult your local Krishi Vigyan Kendra (KVK) for on-field verification.",
                "language": "marathi" if is_marathi else "english",
                "powered_by": f"Google Gemini ({model})",
            }

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
        """Evidence-grounded fallback synthesized directly from RAG documents with intelligent intent matching."""
        # Extract evidence content
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

        # Intelligent Intent Matching for Offline Direct Answer
        if question and question.strip():
            q_clean = question.strip()
            q_lower = q_clean.lower()

            is_where = any(k in q_lower for k in ["where", "occur", "location", "place", "part", "foliage", "leaf", "कुठे", "कोणत्या भागावर", "स्थान"])
            is_what = any(k in q_lower for k in ["what", "meaning", "definition", "identify", "which disease", "काय", "काय आहे", "रोग कोणता"])
            is_why = any(k in q_lower for k in ["why", "cause", "reason", "pathogen", "का", "कशामुळे", "कारण"])
            is_symptom = any(k in q_lower for k in ["symptom", "sign", "look", "spot", "लक्षण", "दिसते कसे", "डाग"])
            is_when = any(k in q_lower for k in ["when", "season", "time", "month", "कधी", "केव्हा", "हंगाम", "वेळ"])
            is_neem = any(k in q_lower for k in ["neem", "organic", "bio", "कडुनिंब", "सेंद्रिय", "जैविक"])
            is_spray = any(k in q_lower for k in ["spray", "fungicide", "pesticide", "chemical", "medicine", "dose", "फवारणी", "औषध", "कीटकनाशक", "बुरशीनाशक"])
            is_water = any(k in q_lower for k in ["water", "irrigation", "moisture", "पाणी", "सिंचन", "ओलावा"])
            is_harvest = any(k in q_lower for k in ["harvest", "fruit", "market", "eat", "phi", "waiting", "काढणी", "फळ", "खाण्यायोग्य", "तोडणी"])

            if is_marathi:
                if is_where:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग प्रामुख्याने पिकाच्या जुन्या व खालच्या पानांवर प्रथम "
                        f"लहान गोलसर तपकिरी डागांच्या स्वरूपात सुरू होतो. रोग वाढल्यास तो खोडावर आणि शेवटी "
                        f"{'बटाट्याच्या कंदांवर' if 'बटाटा' in crop or 'potato' in crop.lower() else 'फळांवर'} पसरतो. "
                        f"हवेतील उष्ण व दमट वातावरणामुळे रोगाचा प्रसार जमिनीकडून वरील शेंड्याकडे होतो."
                    )
                elif is_what:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop} वरील '{disease}' हा एक बुरशीजन्य रोग आहे. "
                        f"यामुळे पानांवर गोलाकार वलयी (concentric rings) असलेले गडद तपकिरी डाग पडतात. "
                        f"पाने पिवळी पडून गळतात, ज्यामुळे प्रकाशसंश्लेषण कमी होऊन उत्पादनात मोठी घट होते."
                    )
                elif is_why:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग 'अल्टरनेरिया' (Alternaria) बुरशीच्या प्रादुर्भावामुळे होतो. "
                        f"सध्याचे तापमान ({weather.get('temperature_c', 25)}°C) आणि हवेतील आर्द्रता ({weather.get('humidity_percentage', 60)}%) "
                        f"या बुरशीच्या बीजाणूंची वाढ वेगाने होण्यासाठी अत्यंत अनुकूल ठरते."
                    )
                elif is_symptom:
                    symp = symptoms_points[0] if symptoms_points else "पानांवर गोल काळे-तपकिरी डाग व कडेने पिवळसर वलय दिसणे."
                    direct_ans = f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop} {disease} ची मुख्य लक्षणे: {symp}"
                elif is_when:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): हा रोग प्रामुख्याने पीक फुलोऱ्यात असताना किंवा पक्व होत असताना "
                        f"ढगाळ आणि दमट हवामानात वेगाने उद्भवतो. सकाळच्या वेळी पानांवर दव साचल्यास प्रादुर्भाव वाढतो."
                    )
                elif is_neem:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): होय, सेंद्रिय नियंत्रणासाठी ५% निंबोळी अर्क (Neem seed kernel extract) "
                        f"किंवा १०,००० ppm निंबोळी तेल २-३ मिली/लिटर पाण्यात मिसळून फवारणी करणे फायदेशीर ठरते. रोग जास्त असल्यास "
                        f"ICAR शिफारशीनुसार प्रमाणित बुरशीनाशकाचा वापर करा."
                    )
                elif is_spray:
                    best_spray = mgmt_points[0] if mgmt_points else "मॅन्कोझेब ७५% WP (२.५ ग्रॅम/लिटर) किंवा कॉपर ऑक्सिक्लोराईड (२.५ ग्रॅम/लिटर)"
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): आयसीएआर (ICAR) शिफारशीनुसार, {crop} वरील {disease} च्या नियंत्रणासाठी "
                        f"मुख्य फवारणी: {best_spray}. फवारणी स्वच्छ सूर्यप्रकाश असताना किंवा सकाळी करावी."
                    )
                elif is_water:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): पाणी देताना झाडांच्या पानांवर पाणी साचणार नाही याची काळजी घ्या. "
                        f"ठिबक सिंचनाचा वापर करा. हवेत आर्द्रता {weather.get('humidity_percentage', 60)}% असल्याने अतिपाणी देणे टाळा."
                    )
                elif is_harvest:
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): फवारणीनंतर रासायनिक औषधांचा प्रतीक्षा काळ (Pre-harvest interval) "
                        f"किमान ७ ते १० दिवस पाळावा. तोपर्यंत फळांची तोडणी करू नये."
                    )
                else:
                    ref_point = mgmt_points[0] if mgmt_points else "रोगट पाने त्वरित काढून टाका व तज्ज्ञांच्या सल्ल्याने फवारणी करा."
                    direct_ans = (
                        f"तुमच्या प्रश्नासाठी ('{q_clean}'): {crop} {disease} बाबत आयसीएआर शिफारस अशी आहे: {ref_point} "
                        f"अधिक माहितीसाठी स्थानिक कृषी विज्ञान केंद्राशी संपर्क साधा."
                    )
            else:
                if is_where:
                    plant_parts = "tubers during harvest" if "potato" in crop.lower() else "fruits and stems"
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): This disease occurs primarily on the older, lower leaves "
                        f"close to the ground first, appearing as circular brown necrotic spots with target-board concentric rings. "
                        f"As the infection advances, it spreads upward through the foliage canopy, petioles, and eventually into the {plant_parts}. "
                        f"Splashing water and humid microclimates accelerate its vertical progression on the plant."
                    )
                elif is_what:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): {crop} {disease} is a destructive fungal condition "
                        f"caused by the pathogen Alternaria. It forms characteristic target-pattern concentric rings on foliage, "
                        f"causing leaf chlorosis (yellowing), early defoliation, and significant yield loss if left unmanaged."
                    )
                elif is_why:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): This occurs due to fungal spore infection (Alternaria species) "
                        f"favored by warm temperatures ({weather.get('temperature_c', 25)}°C) and alternating wet and dry conditions with "
                        f"high ambient humidity ({weather.get('humidity_percentage', 60)}%). Spores overwinter in plant debris and soil."
                    )
                elif is_symptom:
                    symp = symptoms_points[0] if symptoms_points else "Characteristic concentric dark-brown target rings on older foliage with yellow chlorotic halos."
                    direct_ans = f"In response to your query ('{q_clean}'): Primary symptoms of {crop} {disease}: {symp}"
                elif is_when:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Early blight typically appears mid-season as the crop enters tuber bulking or fruit setting, "
                        f"especially when frequent morning dew or overhead irrigation keeps leaf surfaces wet for extended periods."
                    )
                elif is_neem:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Yes, 5% Neem Seed Kernel Extract (NSKE) or neem oil "
                        f"(10,000 ppm @ 2-3 ml/L) is effective as an eco-friendly preventive spray for {crop} against early disease progression. "
                        f"However, if foliar infection exceeds 10-15%, follow up with ICAR-recommended targeted fungicides."
                    )
                elif is_spray:
                    best_spray = mgmt_points[0] if mgmt_points else "Mancozeb 75% WP @ 2.0-2.5 g/L or Chlorothalonil 75% WP @ 2.0 g/L"
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): For {crop} affected by {disease}, the ICAR-recommended primary spray is: "
                        f"{best_spray}. Spray during clear weather in morning hours for optimal leaf absorption."
                    )
                elif is_water:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Avoid overhead sprinkler irrigation which splashes fungal spores onto healthy leaves. "
                        f"Use drip irrigation and maintain soil moisture without waterlogging, especially with current humidity at {weather.get('humidity_percentage', 60)}%."
                    )
                elif is_harvest:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Always observe the mandatory pre-harvest interval (PHI) of 7-14 days after "
                        f"applying any chemical fungicide before picking fruits or lifting tubers for market consumption."
                    )
                else:
                    ref_point = mgmt_points[0] if mgmt_points else "Prune infected lower foliage and maintain proper canopy aeration."
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Based on ICAR research for {crop} {disease}: {ref_point}. "
                        f"Ensure treatments are applied uniformly across the crop canopy."
                    )
        else:
            if is_marathi:
                direct_ans = (
                    f"तात्काळ उपाययोजना ({crop} {disease}): बाधित पाने गोळा करून शेताबाहेर नष्ट करा आणि "
                    f"हवामानातील आर्द्रता लक्षात घेऊन शिफारशीत संरक्षक बुरशीनाशकाची फवारणी करा."
                )
            else:
                direct_ans = (
                    f"Immediate Priority Action for {crop} ({disease}): Remove and destroy infected lower foliage immediately, "
                    f"and apply protective fungicide spray considering prevailing weather conditions."
                )

        if is_marathi:
            explanation = (
                f"हे पीक '{crop}' असून इमेज क्लासिफायर मॉडेलने '{disease}' चे निदान केले आहे "
                f"(विश्वासार्हता: {confidence * 100:.1f}%). "
            )
            if symptoms_points:
                explanation += f"लक्षणे: {symptoms_points[0]}"

            weather_interp = (
                f"स्थानिक हवामान स्थिती: तापमान {weather.get('temperature_c', 25)}°C, "
                f"आर्द्रता {weather.get('humidity_percentage', 60)}%. {weather_note}"
            )

            management = (
                "कृषी तज्ज्ञ शिफारशीनुसार उपाययोजना:\n" +
                ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else
                 "• प्रादुर्भावग्रस्त पाने तोडून नष्ट करा.\n• ठिबक सिंचनाचा वापर करा आणि पानांवर पाणी साचू देऊ नका.\n• प्रमाणित बुरशीनाशक/कीटकनाशकाचा योग्य प्रमाणात वापर करा.")
            )

            prevention = (
                "भविष्यातील प्रतिबंधात्मक उपाय:\n"
                "• पिकांची योग्य फेरपालट (Crop Rotation) करा.\n"
                "• शेतात हवा खेळती राहण्यासाठी योग्य अंतर ठेवा.\n"
                "• ट्रायकोडर्मा किंवा सेंद्रिय खतांचा वापर करून जमिनीचे आरोग्य सुधारा."
            )

            precautions = "फवारणी करताना संरक्षक मास्क व हातमोजे वापरा. काढणीपूर्वी रासायनिक प्रतीक्षेत काळ (Waiting Period) पाळा."
            warning = (
                f"सूचना: मॉडेलची विश्वासार्हता कमी ({confidence * 100:.1f}%) आहे. प्रत्यक्ष शेतात तज्ज्ञांकडून खात्री करूनच निर्णय घ्या."
                if is_low_confidence else ""
            )
            advisory = "मोठ्या प्रमाणात फवारणी करण्यापूर्वी आपल्या जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा तालुका कृषी अधिकाऱ्यांचा सल्ला घ्या."

        else:
            explanation = (
                f"The leaf was classified as {crop} with '{disease}' by the MobileNetV3 vision model "
                f"with {confidence * 100:.1f}% confidence. "
            )
            if symptoms_points:
                explanation += f"Typical symptoms: {symptoms_points[0]}"

            weather_interp = (
                f"Local weather context: {weather.get('location', 'Field')}, "
                f"Temperature: {weather.get('temperature_c', 25)}°C, Humidity: {weather.get('humidity_percentage', 60)}%. "
                f"{weather_note}"
            )

            management = (
                "Recommended Management (from Agricultural Knowledge Base):\n" +
                ("\n".join([f"• {pt}" for pt in mgmt_points[:4]]) if mgmt_points else
                 "• Remove and safely dispose of severely infected foliage.\n• Ensure proper vine aeration and avoid overhead wetting.\n• Apply recommended protective fungicides or bio-agents at labelled rates.")
            )

            prevention = (
                "Preventive Measures for Long-term Protection:\n"
                "• Practice 2-3 season crop rotation with non-host crops.\n"
                "• Maintain optimal plant spacing and stake indeterminate vines.\n"
                "• Use certified disease-free seeds and disease-resistant hybrids."
            )

            precautions = "Always wear protective gear during spray operations. Observe required pre-harvest withholding intervals."
            warning = (
                f"Caution: Model prediction confidence is low ({confidence * 100:.1f}%). Symptoms may be non-typical or in early stages. Upload a clearer image or consult an expert."
                if is_low_confidence else ""
            )
            advisory = "Consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer before chemical treatments."

        # Extract soil context if available
        st_name = "Regional Soil"
        m_val = None
        t_val = 25
        ph_val = "7.0"
        if soil and isinstance(soil, dict):
            st = soil.get("soil_type", {})
            if isinstance(st, dict):
                st_name = st.get("name") or st.get("order") or "Regional Black/Loam Soil"
            elif isinstance(st, str):
                st_name = st
            else:
                st_name = "Regional Black/Loam Soil"

            params = soil.get("soil_parameters", {})
            if isinstance(params, dict):
                rz = params.get("soil_moisture_root_zone", {}) if isinstance(params.get("soil_moisture_root_zone"), dict) else {}
                m_val = rz.get("value")
                stemp = params.get("soil_temperature_surface", {}) if isinstance(params.get("soil_temperature_surface"), dict) else {}
                t_val = stemp.get("value", 25)
                sph = params.get("soil_ph", {}) if isinstance(params.get("soil_ph"), dict) else {}
                ph_val = sph.get("value", "7.0")

        if is_marathi:
            soil_interp = (
                f"स्थानिक माती आरोग्य विश्लेषण: {st_name} जमिनीत मुळांच्या थरातील ओलावा {m_val if m_val is not None else 'योग्य'} m³/m³ आणि तापमान {t_val}°C आहे (सामू: {ph_val}). "
                f"जमिनीतील ओलावा आणि वातावरणातील दमट हवेचा समतोल राखण्यासाठी शेतात पाणी साचू न देणे आणि वापसा स्थितीवरच पाणी देणे आवश्यक आहे."
            )
        else:
            soil_interp = (
                f"Live Soil Health Analysis: {st_name} exhibits root-zone moisture of {m_val if m_val is not None else 'favorable'} m³/m³ and soil temperature of {t_val}°C (typical pH: {ph_val}). "
                f"Prevailing soil moisture combined with ambient humidity governs root aeration and fungal spore viability. Maintain adequate drainage and avoid waterlogging."
            )

        return {
            "direct_answer": direct_ans,
            "explanation": explanation,
            "weather_interpretation": weather_interp,
            "soil_interpretation": soil_interp,
            "management_guidance": management,
            "prevention": prevention,
            "precautions": precautions,
            "uncertainty_warning": warning,
            "expert_advisory": advisory,
            "language": "marathi" if is_marathi else "english",
            "powered_by": "Curated ICAR Knowledge Base (Offline RAG Engine)",
        }



    def generate_soil_guidance(
        self,
        location: dict[str, Any],
        soil_parameters: dict[str, Any],
        data_quality: dict[str, Any],
        crop: str | None,
        retrieved_evidence: list[dict[str, Any]],
        language: str = "english",
        weather_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate evidence-grounded soil health guidance using Gemini with multi-model fallback."""
        is_marathi = (language or "").strip().lower() == "marathi"
        client = self._get_client()

        if client:
            for model in DEFAULT_MODELS:
                try:
                    result = self._call_gemini_soil(
                        client=client,
                        model=model,
                        location=location,
                        soil_parameters=soil_parameters,
                        data_quality=data_quality,
                        crop=crop,
                        evidence=retrieved_evidence,
                        is_marathi=is_marathi,
                        weather=weather_context or {},
                    )
                    if result:
                        return result
                except Exception as e:
                    print(f"Gemini soil model '{model}' failed: {e}. Trying next model...")

        return self._fallback_soil_guidance(
            location=location,
            soil_parameters=soil_parameters,
            data_quality=data_quality,
            crop=crop,
            evidence=retrieved_evidence,
            is_marathi=is_marathi,
            weather=weather_context or {},
        )

    def _call_gemini_soil(
        self,
        client: Any,
        model: str,
        location: dict[str, Any],
        soil_parameters: dict[str, Any],
        data_quality: dict[str, Any],
        crop: str | None,
        evidence: list[dict[str, Any]],
        is_marathi: bool,
        weather: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Construct prompt, invoke Gemini, and parse JSON for soil guidance."""
        prompt = self._build_soil_prompt(
            location, soil_parameters, data_quality, crop, evidence, is_marathi, weather
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
            parsed["language"] = "marathi" if is_marathi else "english"
            parsed["powered_by"] = f"Google Gemini ({model})"
            return parsed
        except Exception as e:
            print(f"Failed to parse Gemini soil JSON output: {e}")
            return None

    def _build_soil_prompt(
        self,
        location: dict[str, Any],
        soil_parameters: dict[str, Any],
        data_quality: dict[str, Any],
        crop: str | None,
        evidence: list[dict[str, Any]],
        is_marathi: bool,
        weather: dict[str, Any],
    ) -> str:
        district = location.get("district", "Maharashtra")
        state = location.get("state", "Maharashtra")
        target_crop = crop or "General Horticultural Crops"
        missing_params = data_quality.get("unavailable_parameters", [])

        # Format parameters
        params_summary = []
        for k, v in soil_parameters.items():
            if isinstance(v, dict):
                p_name = v.get("display_name", k)
                val = v.get("value")
                unit = v.get("unit") or ""
                cat = v.get("data_category", "estimate")
                interp = v.get("interpretation", "")
                params_summary.append(f"- {p_name}: {val} {unit} [{cat}] -> {interp}")

        evidence_text = "\n\n".join([
            f"Source: {e.get('source')} | Title: {e.get('title')}\nSection: {e.get('heading')}\n{e.get('content')}"
            for e in evidence
        ])

        lang_instruction = (
            "LANGUAGE: Generate the entire JSON response in clear, respectful, natural agricultural Marathi (मराठी). "
            "Use standard Marathi farming terms (उदा. सेंद्रिय कर्ब, सामू (pH), पाण्याचा निचरा, शेणखत, माती परीक्षण)."
            if is_marathi
            else "LANGUAGE: Generate the entire JSON response in professional, accessible English."
        )

        prompt = f"""You are an expert Agronomist and Soil Health Scientist advising an Indian farmer in {district}, {state}.
Target Crop: {target_crop}.

SOIL DATA RETRIEVED FROM VERIFIED SOURCES:
{'\n'.join(params_summary) if params_summary else 'No direct parameter values retrieved.'}

CURRENT WEATHER & MICROCLIMATE CONTEXT:
Temperature: {weather.get('temperature_c', 'N/A')}°C, Humidity: {weather.get('humidity_percentage', 'N/A')}%, Condition: {weather.get('condition', 'Seasonal')}

DATA INTEGRITY & SCIENTIFIC ETHICS CONSTRAINTS:
1. Distinguish between actual measurements and regional estimates. Any value tagged as 'regional_estimate' or 'location_measured' is a broad regional or 250m gridded estimate, NOT a farm-specific measurement.
2. The following parameters were explicitly NOT FOUND in public databases: {', '.join(missing_params)}. NEVER invent, fabricate, or assume values for these parameters.
3. DO NOT prescribe precise chemical fertilizer doses (e.g. 'apply exactly 120 kg Urea') based on missing or estimated values.
4. Base recommendations on verified ICAR and TNAU agronomic guidelines from the retrieved evidence.
5. Provide actionable guidance on:
   - Soil reaction / pH implications for {target_crop}.
   - Topsoil and root-zone soil moisture management.
   - Soil Organic Carbon (SOC) enhancement via Farmyard Manure (FYM), compost, green manuring.
   - Soil aeration, drainage, and compaction avoidance.
   - Explicit recommendation on when and how to conduct a physical laboratory Soil Health Card test at a local KVK or Soil Testing Lab.

RETRIEVED ICAR & TNAU RESEARCH EVIDENCE:
{evidence_text if evidence_text else 'Standard ICAR soil management protocols.'}

{lang_instruction}

Respond with a strictly valid JSON object matching this schema:
{{
  "summary": "High-level summary of soil condition, data availability, and overall outlook.",
  "soil_condition_overview": "Explanation of available soil properties (pH, moisture, texture, organic matter) and their agronomic meaning.",
  "crop_suitability": "Evaluation of whether this soil environment is favorable for {target_crop} and what precautions are needed.",
  "recommendations": [
    "Practical recommendation 1 (organic matter/drainage/soil prep)",
    "Practical recommendation 2 (moisture/fertigation management)",
    "Practical recommendation 3 (specific management practice)"
  ],
  "organic_amendments": [
    "Well-decomposed Farmyard manure (FYM) or compost dosage guidelines",
    "Green manuring or bio-fertilizer recommendation (Azotobacter, PSB, Trichoderma)"
  ],
  "irrigation_advice": "Actionable watering and drainage advice based on root-zone moisture.",
  "soil_testing_advice": "Detailed guidance on collecting a composite soil sample for lab analysis at nearest KVK or STL.",
  "limitations_notice": "Clear explanation that regional estimates and satellite moisture provide general guidance and cannot replace a laboratory Soil Health Card test."
}}

Return ONLY valid JSON. Do not include markdown ticks like ```json."""
        return prompt

    def _fallback_soil_guidance(
        self,
        location: dict[str, Any],
        soil_parameters: dict[str, Any],
        data_quality: dict[str, Any],
        crop: str | None,
        evidence: list[dict[str, Any]],
        is_marathi: bool,
        weather: dict[str, Any],
    ) -> dict[str, Any]:
        """Evidence-grounded offline fallback synthesized from ICAR research documents."""
        district = location.get("district", "Maharashtra")
        target_crop = crop or "Horticultural Crops"

        # Extract values
        ph_val = soil_parameters.get("ph", {}).get("value")
        moist_val = soil_parameters.get("soil_moisture", {}).get("value")
        soc_val = soil_parameters.get("organic_carbon", {}).get("value")
        soil_type = soil_parameters.get("soil_type", {}).get("value", "Medium Black Soils (Vertic Inceptisols)")

        if is_marathi:
            summary = (
                f"{district} जिल्ह्यातील उपलब्ध प्रादेशिक माती माहितीनुसार, येथील जमीन प्रामुख्याने {soil_type} प्रकारची आहे. "
                "उपलब्ध माहिती प्रादेशिक अंदाज आणि उपग्रह नोंदींवर आधारित आहे; हे थेट शेतातील प्रयोगशाळा तपासणीचे परिणाम नाहीत."
            )
            overview = (
                f"मातीचा सामू (pH): {ph_val if ph_val else 'माहिती उपलब्ध नाही'} | "
                f"सेंद्रिय कर्ब: {soc_val if soc_val else 'माहिती उपलब्ध नाही'}% | "
                f"मुळांमधील ओलावा: {moist_val if moist_val else 'माहिती उपलब्ध नाही'}%. "
                "दख्खनच्या पठारावरील भारी काळ्या जमिनीत ओलावा धरून ठेवण्याची क्षमता जास्त असते, परंतु अति पावसात निचरा आवश्यक ठरतो."
            )
            suitability = (
                f"{target_crop} पिकासाठी उत्तम निचरा असलेली, भुसभुशीत आणि सेंद्रिय कर्बयुक्त जमीन आवश्यक असते. "
                f"सामू (pH) ६.० ते ७.५ दरम्यान असल्यास अन्नद्रव्ये पिकाला सहज उपलब्ध होतात."
            )
            recommendations = [
                "जमिनीची सुपीकता टिकवण्यासाठी एकरी किमान ५ ते ७ टन चांगले कुजलेले शेणखत किंवा २ टन गांडूळ खत मिसळावे.",
                "ताग किंवा धैंचा यासारखी हिरवळीची खते जमिनीत गाडून सेंद्रिय कर्बाचे प्रमाण वाढवावे.",
                "भारी काळ्या जमिनीत पाणी साचू नये म्हणून गादी वाफे (Raised beds) पद्धतीचा वापर करावा.",
                "रासायनिक खतांचा असंतुलित वापर टाळावा आणि नत्र, स्फुरद, पालाश यांचा समतोल राखावा.",
            ]
            amendments = [
                "चांगले कुजलेले शेणखत (FYM) किंवा कंपोस्ट १०-१५ टन/हेक्टर पूर्वमशागतीवेळी जमिनीत मिसळावे.",
                "ऍझोटोबॅक्टर आणि स्फुरद विरघळवणारे जिवाणू संवर्धन (PSB) प्रति एकर २ किलो शेणखतात मिसळून द्यावे.",
            ]
            irrigation = (
                f"सध्या मुळांच्या भागातील ओलावा {moist_val if moist_val else 'समाधानकारक'}% आहे. "
                "भारी जमिनीत पाणी साचू न देता वाफसा स्थितीत ठिबक सिंचनाने पाणी द्यावे."
            )
            testing = (
                "नजीकच्या कृषी विज्ञान केंद्र (KVK) किंवा मान्यताप्राप्त माती परीक्षण प्रयोगशाळेत आपल्या शेतातील मातीचे "
                "प्रातिनिधिक नमुने तपासून अधिकृत 'सॉईल हेल्थ कार्ड' (Soil Health Card) प्राप्त करून घ्यावे."
            )
            limitations = (
                "ही माहिती ICAR-NBSS&LUP प्रादेशिक सर्वेक्षण आणि उपग्रह नोंदींवर आधारित आहे. "
                "प्रत्येक शेतातील खतांच्या पूर्वीच्या वापरामुळे प्रत्यक्षात फरक असू शकतो, त्यामुळे प्रत्यक्ष माती परीक्षण आवश्यक आहे."
            )
        else:
            summary = (
                f"Based on regional soil survey and remote-sensing data for {district}, predominant soil is {soil_type}. "
                "Values reflect regional estimates and gridded models, not individual farm measurements."
            )
            overview = (
                f"Soil pH: {ph_val if ph_val else 'Not Available'} | "
                f"Soil Organic Carbon: {soc_val if soc_val else 'Not Available'}% | "
                f"Root-zone Volumetric Moisture: {moist_val if moist_val else 'Not Available'}%. "
                "Deccan Vertisols possess high clay content and moisture retention, requiring good drainage management."
            )
            suitability = (
                f"Target crop '{target_crop}' requires well-aerated, friable soil with good drainage and balanced organic matter. "
                f"Optimal pH range between 6.0 and 7.2 supports maximum macro and micronutrient uptake."
            )
            recommendations = [
                "Incorporate 10-15 tonnes/ha of well-decomposed Farmyard Manure (FYM) or 2.5-5 t/ha vermicompost before planting.",
                "Practice in-situ green manuring with Sunnhemp or Dhaincha to enrich topsoil organic carbon and microbial activity.",
                "Adopt raised bed planting with drip fertigation on heavy clay soils to avoid root waterlogging and collar rot.",
                "Avoid indiscriminate chemical fertilizer application without verified laboratory test figures.",
            ]
            amendments = [
                "Farmyard Manure (FYM) 10-15 t/ha or well-cured compost incorporated during basal land preparation.",
                "Inoculate soil with bio-fertilizers: Azotobacter/Azospirillum and Phosphate Solubilizing Bacteria (PSB) at 5 kg/ha with FYM.",
            ]
            irrigation = (
                f"Current root-zone soil moisture is approximately {moist_val if moist_val else 'moderate'}%. "
                "Maintain uniform field capacity and avoid prolonged standing water around plant root collars."
            )
            testing = (
                "Farmers are strongly advised to submit a representative composite field soil sample to an authorized "
                "Soil Testing Laboratory (STL) or Krishi Vigyan Kendra (KVK) to obtain a certified Soil Health Card."
            )
            limitations = (
                "This assessment is based on public regional databases and gridded land surface models. "
                "It serves as a regional baseline and does not replace laboratory physical soil testing."
            )

        return {
            "summary": summary,
            "soil_condition_overview": overview,
            "crop_suitability": suitability,
            "recommendations": recommendations,
            "organic_amendments": amendments,
            "irrigation_advice": irrigation,
            "soil_testing_advice": testing,
            "limitations_notice": limitations,
            "language": "marathi" if is_marathi else "english",
            "powered_by": "Curated ICAR Soil Knowledge Base (Offline RAG Engine)",
        }


# Singleton getter
_default_gemini = None


def get_gemini_service() -> GeminiGuidanceService:
    global _default_gemini
    if _default_gemini is None:
        _default_gemini = GeminiGuidanceService()
    return _default_gemini
