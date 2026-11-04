"""Agricultural localization and multilingual translation tables.

Provides shared English/Marathi translation dictionaries and weather localization
enrichment for crops, diseases, Maharashtra cities/talukas, and meteorological conditions.
Used across backend API endpoints, LangGraph nodes, and formatting utilities.
"""
from __future__ import annotations

from typing import Any

# --- Comprehensive Multilingual Localization Tables ---
CROP_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Potato": {"en": "Potato", "mr": "बटाटा"},
    "Tomato": {"en": "Tomato", "mr": "टोमॅटो"},
    "Pepper, bell": {"en": "Bell pepper", "mr": "ढोबळी मिरची"},
    "Bell pepper": {"en": "Bell pepper", "mr": "ढोबळी मिरची"},
    "General / Solanaceae": {"en": "General Crop", "mr": "सर्व पिके / शेती सल्ला"},
    "General": {"en": "General Crop", "mr": "सर्व पिके / शेती सल्ला"},
}

DISEASE_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Early blight": {"en": "Early blight", "mr": "लवकर येणारा करपा (Early Blight)"},
    "Late blight": {"en": "Late blight", "mr": "उशिरा येणारा करपा (Late Blight)"},
    "Bacterial spot": {"en": "Bacterial spot", "mr": "जिवाणूजन्य ठिपके (Bacterial Spot)"},
    "Leaf Mold": {"en": "Leaf Mold", "mr": "पानावरील बुरशी (Leaf Mold)"},
    "Septoria leaf spot": {"en": "Septoria leaf spot", "mr": "सेप्टोरिया पानावरील ठिपके (Septoria Leaf Spot)"},
    "Spider mites Two-spotted spider mite": {"en": "Two-spotted spider mite", "mr": "दोन ठिपक्यांची लाल कोळी (Spider Mites)"},
    "Target Spot": {"en": "Target Spot", "mr": "लक्ष्य ठिपके (Target Spot)"},
    "Tomato Yellow Leaf Curl Virus": {"en": "Yellow Leaf Curl Virus", "mr": "पिवळा चुरडा-मुरडा विषाणू (TYLCV)"},
    "Tomato mosaic virus": {"en": "Tomato mosaic virus", "mr": "टोमॅटो मोझॅक विषाणू (Mosaic Virus)"},
    "healthy": {"en": "Healthy Crop", "mr": "निरोगी पीक (Healthy)"},
    "Healthy": {"en": "Healthy Crop", "mr": "निरोगी पीक (Healthy)"},
    "General Agronomic Inquiry": {"en": "Expert Q&A Guidance", "mr": "कृषी तज्ज्ञ सल्ला व मार्गदर्शन"},
}

CITY_TRANSLATIONS_MR: dict[str, str] = {
    "pune": "पुणे",
    "akurdi": "आकुर्डी",
    "nashik": "नाशिक",
    "nagpur": "नागपूर",
    "aurangabad": "छत्रपती संभाजीनगर",
    "chhatrapati sambhajinagar": "छत्रपती संभाजीनगर",
    "solapur": "सोलापूर",
    "kolhapur": "कोल्हापूर",
    "satara": "सातारा",
    "sangli": "सांगली",
    "jalgaon": "जळगाव",
    "ahmednagar": "अहमदनगर",
    "amravati": "अमरावती",
    "nanded": "नांदेड",
    "mumbai": "मुंबई",
    "thane": "ठाणे",
    "baramati": "बारामती",
}

WEATHER_CONDITIONS_MR: dict[str, str] = {
    "clear": "स्वच्छ आकाश",
    "sunny": "सूर्यप्रकाश",
    "partly cloudy": "अंशतः ढगाळ",
    "cloudy": "ढगाळ हवामान",
    "overcast": "पूर्ण ढगाळ",
    "rain": "पाऊस",
    "light rain": "हलका पाऊस",
    "moderate rain": "मध्यम पाऊस",
    "heavy rain": "मुसळधार पाऊस",
    "thunderstorm": "वादळी पाऊस",
    "drizzle": "रिमझिम पाऊस",
    "fog": "धुके",
    "mist": "धुके",
    "haze": "धूसर",
    "normal": "सामान्य हवामान",
    "seasonal": "हंगामी हवामान",
}


def localize_weather_data(weather: dict[str, Any]) -> dict[str, Any]:
    """Enrich weather dictionary with Marathi localization fields."""
    if not isinstance(weather, dict):
        return weather
    w = dict(weather)
    loc = str(w.get("location", "")).lower()
    city_mr = None
    for k, v in CITY_TRANSLATIONS_MR.items():
        if k in loc:
            city_mr = f"{v}, भारत"
            break
    w["location_mr"] = city_mr or f"{w.get('location', 'शेत')}"

    cond = str(w.get("condition", "")).lower()
    w["condition_mr"] = WEATHER_CONDITIONS_MR.get(
        cond, WEATHER_CONDITIONS_MR.get(cond.strip(), w.get("condition", "हंगामी हवामान"))
    )

    risk = str(w.get("risk_analysis", ""))
    hum = w.get("humidity_percentage", 60)
    if "high humidity" in risk.lower() or "humidity" in risk.lower():
        w["risk_analysis_mr"] = (
            f"हवेतील जास्त आर्द्रता ({hum}%) बुरशीच्या बीजाणूंचा प्रसार वेगाने होण्यास अनुकूल आहे. शेतात पाण्याचा निचरा योग्य ठेवा."
        )
    elif "rain" in risk.lower():
        w["risk_analysis_mr"] = "पावसाची शक्यता असल्याने फवारणी करताना स्टीकर (Sticker) वापरा किंवा पाऊस थांबल्यावर फवारा."
    else:
        w["risk_analysis_mr"] = "सध्याचे स्थानिक हवामान सामान्य असून नियमित निरीक्षण व प्रतिबंधात्मक काळजी ठेवा."
    return w

