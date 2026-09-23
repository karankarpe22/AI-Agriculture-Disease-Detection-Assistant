"""Phase 10 & 13: Google Gemini Generative AI Guidance Layer with Marathi & English support."""
from __future__ import annotations

import json
import os
import re
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class GeminiGuidanceService:
    """Provides evidence-grounded, contextual agricultural guidance using Google Gemini."""

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name = model_name
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
    ) -> dict[str, Any]:
        """Generate structured agricultural guidance based strictly on inputs."""
        lang = (language or "english").lower().strip()
        is_marathi = lang in ["marathi", "mr", "मराठी"]

        # If Gemini API key is available, call official Google GenAI SDK
        client = self._get_client()
        if client:
            try:
                guidance = self._call_gemini(
                    client=client,
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
                    return guidance
            except Exception as e:
                print(f"Gemini generation call failed ({e}). Falling back to grounded RAG formatter.")

        # Robust Evidence-Grounded Fallback Engine (No API key or API call failed)
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

        target_lang = "Marathi (मराठी)" if is_marathi else "English"

        has_specific_question = bool(question and question.strip())

        if has_specific_question:
            qa_instruction = f"""CRITICAL Q&A INSTRUCTION:
The farmer asked this SPECIFIC QUESTION: "{question.strip()}"
You MUST provide a direct, comprehensive, empathetic, and highly actionable answer addressing this EXACT question in the "direct_answer" field.
- If the farmer asks whether a specific spray/treatment (e.g. neem oil, chemical, fertilizer, bio-agent) is effective, answer directly whether it is recommended, why or why not, and how/when to use it based on ICAR evidence.
- If the farmer asks about timing, dosage, watering, or pre-harvest waiting periods, provide specific practical figures.
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
- Farmer's Question: {question or 'What should I do to treat and protect my crop?'}

RETRIEVED AGRICULTURAL EVIDENCE:
{evidence_text if evidence_text else 'Standard ICAR Horticultural Practices'}

Please return your response in JSON format with exactly these keys:
{{
  "direct_answer": "Direct, empathetic, and comprehensive answer to the farmer's specific question (or immediate executive recommendation if no question was asked).",
  "explanation": "A simple, clear explanation of what this condition is and why it occurred.",
  "weather_interpretation": "How current temperature and humidity impact this disease or spray timing.",
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
            model=self.model_name,
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
            parsed["powered_by"] = f"Google Gemini ({self.model_name})"

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
            # If JSON parsing failed, package raw text into direct_answer & explanation
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
                "powered_by": f"Google Gemini ({self.model_name})",
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
    ) -> dict[str, Any]:
        """Evidence-grounded fallback synthesized directly from RAG documents."""
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

        # Synthesize direct answer to farmer's question from evidence
        if question and question.strip():
            q_clean = question.strip()
            q_lower = q_clean.lower()
            is_neem = "neem" in q_lower or "कडुनिंब" in q_lower or "organic" in q_lower or "सेंद्रिय" in q_lower
            is_spray = "spray" in q_lower or "फवारणी" in q_lower or "medicine" in q_lower or "औषध" in q_lower
            is_water = "water" in q_lower or "पाणी" in q_lower or "irrigation" in q_lower
            is_harvest = "harvest" in q_lower or "काढणी" in q_lower or "fruit" in q_lower or "फळ" in q_lower

            if is_marathi:
                if is_neem:
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
                if is_neem:
                    direct_ans = (
                        f"In response to your query ('{q_clean}'): Yes, 5% Neem Seed Kernel Extract (NSKE) or neem oil "
                        f"(10,000 ppm @ 2-3 ml/L) is effective as an eco-friendly preventive spray for {crop} against early disease progression. "
                        f"However, if foliar infection exceeds 10-15%, follow up with ICAR-recommended targeted fungicides."
                    )
                elif is_spray:
                    best_spray = mgmt_points[0] if mgmt_points else "Mancozeb 75% WP @ 2.5 g/L or Copper Oxychloride @ 2.5 g/L"
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
                        f"applying any chemical fungicide before picking fruits for market consumption."
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

        return {
            "direct_answer": direct_ans,
            "explanation": explanation,
            "weather_interpretation": weather_interp,
            "management_guidance": management,
            "prevention": prevention,
            "precautions": precautions,
            "uncertainty_warning": warning,
            "expert_advisory": advisory,
            "language": "marathi" if is_marathi else "english",
            "powered_by": "Curated ICAR Knowledge Base (Offline RAG Engine)",
        }


# Singleton getter
_default_gemini = None


def get_gemini_service() -> GeminiGuidanceService:
    global _default_gemini
    if _default_gemini is None:
        _default_gemini = GeminiGuidanceService()
    return _default_gemini
