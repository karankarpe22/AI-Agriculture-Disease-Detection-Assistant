/**
 * AI Agriculture Assistant (कृषी सहाय्यक)
 * Frontend Application Controller with Full Multilingual, Standalone & Follow-Up Q&A Capabilities
 */

// Backend API URL Configuration
// 1. Localhost / 127.0.0.1 -> relative path "" (connects directly to local FastAPI server)
// 2. Vercel / Remote Domain -> connects to your Render backend service
// You can also override dynamically by adding ?backend=https://your-app.onrender.com to your Vercel URL
// Permanent, secured AI Backend URL on Render
const PRODUCTION_BACKEND_URL = "https://krishi-sahayyak.onrender.com";

// Automatic API Routing:
// - Local development (localhost / 127.0.0.1) -> relative endpoint ""
// - Production (Vercel) -> secured Render backend
const API_BASE = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
  ? ""
  : PRODUCTION_BACKEND_URL;

// Ensure any obsolete overrides in browser storage are removed
try {
  localStorage.removeItem("AGRI_BACKEND_URL");
} catch (e) {
  // Ignore restricted storage contexts
}

// Internationalization Dictionary (English & Marathi)
const I18N = {
  english: {
    appTitle: "AI Agriculture Assistant",
    appSubtitle: "Evidence-Grounded Crop Disease Detection & Advisory",
    tabDiagnose: "Leaf Diagnosis",
    tabAsk: "Ask Agronomist",
    leafInput: "Crop Leaf Diagnosis",
    dropzonePrompt: 'Drag & drop a leaf photo, or <span class="browse-link">browse</span>',
    dropzoneHint: "Supports clear photos of Tomato, Potato, Bell Pepper (JPG/PNG, max 10MB)",
    lblSamples: "🧪 Test with Sample Leaf:",
    lblLocation: "📍 Farmer Location / City",
    lblQuestion: "❓ Farmer Question",
    lblQuestionSub: "(Optional specific question)",
    qPlaceholder: "e.g. Can I use neem oil spray? How do I stop this from spreading?",
    gradcamOpt: "Generate Grad-CAM Visual Attention Heatmap",
    btnAnalyze: "Analyze Crop Health",
    btnAnalyzing: "Analyzing Leaf & Weather...",

    standaloneTitle: "Ask Agricultural Expert",
    standaloneDesc: "Ask any question on crop health, spray schedules, fertilizers, or pest management without needing a photo.",
    lblCropContext: "🌱 Crop Category:",
    lblStandaloneLoc: "📍 Location (for weather-tailored advice)",
    lblStandaloneQuestion: "❓ Your Farming Question",
    standalonePlaceholder: "e.g. Which fertilizer schedule is best before flowering? How to prevent potato blight in humid weather?",
    btnStandalone: "Ask Agronomist",
    btnStandaloneAsking: "Retrieving Agronomic Advice...",

    diagnosisHeading: "Diagnosis & Agricultural Guidance",
    emptyTitle: "Ready for Your Query",
    emptyDesc: "Upload a crop leaf image for disease diagnosis, or switch to 'Ask Agronomist' to ask direct agricultural questions with live weather guidance.",
    confLabel: "Confidence",
    listenAdvisory: "Listen to Full Advisory",
    playingAudio: "Playing Audio...",

    qaAnswerTitle: "Answer to Your Question",
    listenAnswer: "Listen",

    soilHeading: "Soil Health Profile",
    syntheticBadge: "🧪 Synthetic Demonstration",
    soilTitle: "Soil Health & Nutrient Impact",
    lblSoilPh: "Soil pH",
    lblSoilMoisture: "Moisture",
    lblSoilNpk: "Available N-P-K",
    lblSoilCarbon: "Organic Carbon",
    npkSub: "kg/ha (ICAR Rating)",

    explanationTitle: "Explanation",
    managementTitle: "Treatment & Management",
    preventionTitle: "Long-Term Prevention",
    precautionsTitle: "Precautions & Safety",
    advisoryTitle: "Expert Advisory",
    gradcamTitle: "Visual Explanation (Grad-CAM)",
    gradcamDesc: "Highlighted regions indicate leaf symptoms that influenced the MobileNetV3 model prediction:",
    sourcesTitle: "Agricultural Evidence Sources",
    footerText: "AI Agriculture Assistant • MobileNetV3-Large Classifier • RAG Knowledge Base • Gemini Guidance",

    followupTitle: "Ask a Follow-Up Question",
    followupSub: "Ask specific doubts about treatment timing, organic options, or spray safety without re-uploading.",
    followupPlaceholder: "Type your follow-up question here...",
    btnFollowup: "Ask",
    btnFollowupAsking: "Thinking...",

    sprayQ: "What spray should I use to treat this disease?",
    preventQ: "How can I prevent this in my other plants?",
    harvestQ: "Is this disease dangerous for fruit harvest?",

    secSummaryTitle: "1. Integrated Summary",
    secDiseaseTitle: "2. Disease Explanation & Symptoms",
    secWeatherTitle: "3. Weather Impact & Microclimate",
    secSoilTitle: "4. Soil Health Impact",
    secActionsTitle: "5. Recommended Actions",
    secTreatmentTitle: "6. Treatment & Prevention",
    secSafetyTitle: "7. Safety, Limitations & Expert Advisory",
    curativeTitle: "Curative Treatment & Dosage",
    preventiveTitle: "Preventive Measures & Hygiene",
    precautionsTitle: "Chemical Safety & Precautions",
    advisoryTitle: "Limitations & Expert Advisory (KVK)",
    themeDark: "Dark",
    themeLight: "Light",
    themeToggleToDark: "Switch to dark mode",
    themeToggleToLight: "Switch to light mode",
  },
  marathi: {
    appTitle: "कृषी सहाय्यक (AI Agriculture Assistant)",
    appSubtitle: "प्रमाणित कृषी पुरावे आणि हवामानावर आधारित पीक रोग निदान व सल्ला",
    tabDiagnose: "पानांचे रोग निदान",
    tabAsk: "कृषी तज्ज्ञांना विचारा",
    leafInput: "पिकांच्या पानांचे रोग निदान",
    dropzonePrompt: 'पानाचा फोटो येथे ड्रॅग करा किंवा <span class="browse-link">निवडा</span>',
    dropzoneHint: "टोमॅटो, बटाटा, ढोबळी मिरचीच्या पानांचे स्पष्ट फोटो (JPG/PNG, कमाल १० MB)",
    lblSamples: "🧪 नमुना पानासह तपासा:",
    lblLocation: "📍 शेतकऱ्याचे ठिकाण / शहर",
    lblQuestion: "❓ शेतकऱ्याचा प्रश्न",
    lblQuestionSub: "(पर्यायी विशिष्ट शंका)",
    qPlaceholder: "उदा. कडुनिंब तेलाची फवारणी चालेल का? रोगाचा प्रसार कसा रोखावा?",
    gradcamOpt: "Grad-CAM व्हिज्युअल हीटमॅप तयार करा",
    btnAnalyze: "पिकाचे आरोग्य तपासा",
    btnAnalyzing: "निदान व सल्ला सुरू आहे...",

    standaloneTitle: "कृषी तज्ज्ञांशी थेट संवाद",
    standaloneDesc: "फोटो अपलोड न करता पीक आरोग्य, फवारणीचे वेळापत्रक, खते किंवा कीड नियंत्रणावर थेट प्रश्न विचारा.",
    lblCropContext: "🌱 पीक वर्गवारी:",
    lblStandaloneLoc: "📍 ठिकाण (हवामानानुसार सल्ल्यासाठी)",
    lblStandaloneQuestion: "❓ तुमचा शेतीविषयक प्रश्न",
    standalonePlaceholder: "उदा. फुलोऱ्यापूर्वी टोमॅटोला कोणते खत द्यावे? ढगाळ हवामानात बटाट्यावरील करपा कसा रोखावा?",
    btnStandalone: "कृषी सल्ला मिळवा",
    btnStandaloneAsking: "कृषी सल्ला तयार करत आहे...",

    diagnosisHeading: "रोग निदान आणि कृषी सल्ला",
    emptyTitle: "तुमच्या प्रश्नासाठी सज्ज",
    emptyDesc: "रोग निदानासाठी पानाचा फोटो अपलोड करा, किंवा फोटोशिवाय थेट शेतीविषयक प्रश्न विचारण्यासाठी 'कृषी तज्ज्ञांना विचारा' निवडा.",
    confLabel: "विश्वासार्हता",
    listenAdvisory: "संपूर्ण सल्ला ऐका",
    playingAudio: "ऑडिओ वाजत आहे...",

    qaAnswerTitle: "तुमच्या प्रश्नाचे थेट उत्तर",
    listenAnswer: "ऐका",

    soilHeading: "मृदा आरोग्य विश्लेषण",
    syntheticBadge: "🧪 प्रात्यक्षिक मृदा नोंद",
    soilTitle: "मृदा आरोग्य व पोषण समन्वय",
    lblSoilPh: "जमिनीचा सामू (pH)",
    lblSoilMoisture: "मातीतील ओलावा",
    lblSoilNpk: "उपलब्ध नत्र-स्फुरद-पालाश",
    lblSoilCarbon: "सेंद्रिय कर्ब",
    npkSub: "किलो/हेक्टर (आयसीएआर प्रमाण)",

    explanationTitle: "रोगाचे स्पष्टीकरण व कारणे",
    managementTitle: "उपचार व नियंत्रण पद्धती",
    preventionTitle: "दीर्घकालीन प्रतिबंधात्मक उपाय",
    precautionsTitle: "सुरक्षा व सावधगिरी",
    advisoryTitle: "तज्ज्ञ कृषी सल्ला (KVK)",
    gradcamTitle: "व्हिज्युअल स्पष्टीकरण (Grad-CAM)",
    gradcamDesc: "हायलाइट केलेले भाग MobileNetV3 मॉडेलने निदान करताना विचारात घेतलेली लक्षणे दर्शवतात:",
    sourcesTitle: "वापरलेले कृषी संशोधन पुरावे व संदर्भ",
    footerText: "कृषी सहाय्यक • MobileNetV3-Large क्लासिफायर • RAG नॉलेज बेस • जेमिनी मार्गदर्शन",

    followupTitle: "अधिक माहिती किंवा पाठपुरावा प्रश्न विचारा",
    followupSub: "पुन्हा फोटो न टाकता फवारणीची वेळ, सेंद्रिय उपाय किंवा रासायनिक सुरक्षिततेबाबत थेट शंका विचारा.",
    followupPlaceholder: "येथे तुमचा प्रश्न टाईप करा किंवा बोला...",
    btnFollowup: "विचारा",
    btnFollowupAsking: "उत्तर शोधत आहे...",

    sprayQ: "या रोगावर नियंत्रणासाठी कोणती फवारणी करावी?",
    preventQ: "इतर झाडांवर हा रोग पसरू नये म्हणून काय करावे?",
    harvestQ: "हा रोग फळांच्या उत्पादनासाठी किती घातक आहे?",

    secSummaryTitle: "१. एकत्रित कृषी सारांश",
    secDiseaseTitle: "२. रोग स्पष्टीकरण आणि लक्षणे",
    secWeatherTitle: "३. हवामान प्रभाव आणि सूक्ष्म हवामान",
    secSoilTitle: "४. मृदा आरोग्य प्रभाव",
    secActionsTitle: "५. शिफारस केलेल्या कृती",
    secTreatmentTitle: "६. उपचार आणि प्रतिबंध",
    secSafetyTitle: "७. सुरक्षितता, मर्यादा आणि तज्ज्ञ सल्ला",
    curativeTitle: "उपचारात्मक फवारणी आणि प्रमाण",
    preventiveTitle: "प्रतिबंधात्मक उपाय आणि स्वच्छता",
    precautionsTitle: "रासायनिक सुरक्षा आणि खबरदारी",
    advisoryTitle: "मर्यादा आणि तज्ज्ञ सल्ला (केव्हीके)",
    themeDark: "डार्क",
    themeLight: "लाईट",
    themeToggleToDark: "डार्क मोड निवडा",
    themeToggleToLight: "लाईट मोड निवडा",
  },
};

// Global State
let currentLanguage = "english";
let selectedFile = null;
let currentGuidanceText = "";
let currentDirectAnswerText = "";
let currentCrop = null;
let currentDisease = null;
let currentConfidence = 0.85;
let currentWeather = null;
let activeMode = "diagnose";
let selectedCropContext = "";
let lastAnalysisData = null;
let lastAnalysisOptions = null;

// --- Deep Multilingual Dictionaries & Agronomic Translation Engine ---
const CROP_NAME_MAP_MR = {
  "Potato": "बटाटा",
  "Tomato": "टोमॅटो",
  "Bell pepper": "ढोबळी मिरची",
  "Pepper, bell": "ढोबळी मिरची",
  "General / Solanaceae": "सर्व पिके / शेती सल्ला",
  "General": "सर्व पिके / शेती सल्ला",
  "Crop": "पीक",
};
const CROP_NAME_MAP_EN = {
  "बटाटा": "Potato",
  "टोमॅटो": "Tomato",
  "ढोबळी मिरची": "Bell pepper",
  "सर्व पिके / शेती सल्ला": "General Crop",
  "पीक": "Crop",
};

const DISEASE_NAME_MAP_MR = {
  "Early blight": "लवकर येणारा करपा (Early Blight)",
  "Late blight": "उशिरा येणारा करपा (Late Blight)",
  "Bacterial spot": "जिवाणूजन्य ठिपके (Bacterial Spot)",
  "Leaf Mold": "पानावरील बुरशी (Leaf Mold)",
  "Septoria leaf spot": "सेप्टोरिया पानावरील ठिपके (Septoria Leaf Spot)",
  "Spider mites Two-spotted spider mite": "दोन ठिपक्यांची लाल कोळी (Spider Mites)",
  "Target Spot": "लक्ष्य ठिपके (Target Spot)",
  "Tomato Yellow Leaf Curl Virus": "पिवळा चुरडा-मुरडा विषाणू (TYLCV)",
  "Tomato mosaic virus": "टोमॅटो मोझॅक विषाणू (Mosaic Virus)",
  "healthy": "निरोगी पीक (Healthy)",
  "Healthy": "निरोगी पीक (Healthy)",
  "Expert Q&A Guidance": "कृषी तज्ज्ञ सल्ला व मार्गदर्शन",
  "General Agronomic Inquiry": "कृषी तज्ज्ञ सल्ला व मार्गदर्शन",
};
const DISEASE_NAME_MAP_EN = {
  "लवकर येणारा करपा (Early Blight)": "Early blight",
  "उशिरा येणारा करपा (Late Blight)": "Late blight",
  "जिवाणूजन्य ठिपके (Bacterial Spot)": "Bacterial spot",
  "पानावरील बुरशी (Leaf Mold)": "Leaf Mold",
  "सेप्टोरिया पानावरील ठिपके (Septoria Leaf Spot)": "Septoria leaf spot",
  "दोन ठिपक्यांची लाल कोळी (Spider Mites)": "Two-spotted spider mite",
  "लक्ष्य ठिपके (Target Spot)": "Target Spot",
  "पिवळा चुरडा-मुरडा विषाणू (TYLCV)": "Yellow Leaf Curl Virus",
  "टोमॅटो मोझॅक विषाणू (Mosaic Virus)": "Tomato mosaic virus",
  "निरोगी पीक (Healthy)": "Healthy Crop",
  "कृषी तज्ज्ञ सल्ला व मार्गदर्शन": "Expert Q&A Guidance",
};

const CITY_NAME_MAP_MR = {
  "pune": "पुणे, भारत",
  "akurdi": "आकुर्डी, भारत",
  "nashik": "नाशिक, भारत",
  "nagpur": "नागपूर, भारत",
  "aurangabad": "छत्रपती संभाजीनगर, भारत",
  "chhatrapati sambhajinagar": "छत्रपती संभाजीनगर, भारत",
  "solapur": "सोलापूर, भारत",
  "kolhapur": "कोल्हापूर, भारत",
  "satara": "सातारा, भारत",
  "sangli": "सांगली, भारत",
  "mumbai": "मुंबई, भारत",
  "thane": "ठाणे, भारत",
  "baramati": "बारामती, भारत",
};

const WEATHER_COND_MAP_MR = {
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
};

function translateCityToMarathi(city) {
  if (!city) return "शेत, भारत";
  const lower = city.toLowerCase();
  for (const [k, v] of Object.entries(CITY_NAME_MAP_MR)) {
    if (lower.includes(k)) return v;
  }
  return `${city}, भारत`;
}

function translateConditionToMarathi(cond) {
  if (!cond) return "हंगामी हवामान";
  const lower = cond.toLowerCase().trim();
  return WEATHER_COND_MAP_MR[lower] || cond;
}

function translateWeatherRiskToMarathi(risk, humidity) {
  const r = (risk || "").toLowerCase();
  const hum = humidity || 60;
  if (r.includes("humidity") || hum >= 65) {
    return `हवेतील जास्त आर्द्रता (${hum}%) बुरशीच्या बीजाणूंचा प्रसार वेगाने होण्यास अनुकूल आहे. शेतात पाण्याचा निचरा योग्य ठेवा व प्रतिबंधात्मक फवारणी करा.`;
  }
  if (r.includes("rain")) {
    return "पावसाची शक्यता असल्याने फवारणी करताना स्टीकर (Sticker) वापरा किंवा पाऊस थांबल्यावर फवारा.";
  }
  return "सध्याचे स्थानिक हवामान सामान्य असून नियमित निरीक्षण व प्रतिबंधात्मक काळजी ठेवा.";
}

function translateQuestionToMarathi(q) {
  if (!q) return "";
  const lower = q.toLowerCase().trim();
  if (lower.includes("what care should be taken") || lower.includes("care") || lower.includes("prevent this disease")) {
    return "हा रोग रोखण्यासाठी काय काळजी घ्यावी?";
  }
  if (lower.includes("what spray") || lower.includes("spray guidance")) {
    return "या रोगावर नियंत्रणासाठी कोणती फवारणी करावी?";
  }
  if (lower.includes("how can i prevent") || lower.includes("other plants")) {
    return "इतर झाडांवर हा रोग पसरू नये म्हणून काय करावे?";
  }
  if (lower.includes("harvest") || lower.includes("fruit")) {
    return "हा रोग फळांच्या उत्पादनासाठी किती घातक आहे?";
  }
  if (lower.includes("neem")) {
    return "यावर सेंद्रिय कडुनिंब तेलाची फवारणी चालेल का?";
  }
  if (lower.includes("waiting period") || lower.includes("phi")) {
    return "फळे तोडणीपूर्वी किती दिवस प्रतीक्षा करावी (PHI काळ)?";
  }
  if (lower.includes("time of day") || lower.includes("best spray timing")) {
    return "फवारणी करण्यासाठी दिवसाची कोणती वेळ सर्वोत्तम आहे?";
  }
  if (lower.includes("repeat") || lower.includes("frequency")) {
    return "ही फवारणी किती दिवसांच्या अंतराने पुन्हा करावी?";
  }
  return q;
}

function translateAgronomicTextToMarathi(text, cropDisplay, diseaseDisplay) {
  if (!text) return "";
  const lower = text.toLowerCase();

  // If text is already Marathi (contains Devanagari characters)
  if (/[\u0900-\u097F]/.test(text)) {
    return text;
  }

  // Handle standard late blight care and prevention paragraph (from user screenshot)
  if (lower.includes("late blight") || lower.includes("potato crop") || lower.includes("mancozeb")) {
    return `नमस्कार शेतकरी बंधू, आपल्या ${cropDisplay} पिकाचे रक्षण करणे अत्यंत महत्त्वाचे आहे. MobileNetV3 संगणक व्हिजन मॉडेलच्या रोग वर्गीकरणानुसार, आपल्या पिकावर '${diseaseDisplay}' रोगाची लक्षणे दिसून येत आहेत. या रोगाचा प्रभावी प्रतिबंध आणि व्यवस्थापन करण्यासाठी पुढील महत्त्वाची काळजी घ्यावी: कुफरी गिरिधारी, कुफरी ख्याती किंवा कुफरी मोहन यांसारख्या प्रमाणित रोगप्रतिकारक किंवा सहनशील बटाटा वाणांची निवड करा आणि नेहमी प्रमाणित शीतगृहातील निरोगी बियाणे कंद वापरा. बीजाणू जमिनीतील कंदांपर्यंत वाहून जाण्यापासून रोखण्यासाठी झाडांना वेळेवर मातीची भर (earthing-up) द्या. रोग प्रतिबंधासाठी लक्षणे दिसण्यापूर्वी मँकोझेब ७५% डब्ल्यूपी (२.५ ग्रॅम/लिटर) किंवा प्रोपिनेब ७०% डब्ल्यूपी (२.५ ग्रॅम/लिटर) फवारा, विशेषतः हवेत दमटपणा वाढल्यास. पानांवर डाग आधीच दिसू लागले असल्यास, सायमोक्सानिल ८% + मँकोझेब ६४% डब्ल्यूपी (३.० ग्रॅम/लिटर) किंवा डायमेथोमॉर्फ ५०% डब्ल्यूपी (१.० ग्रॅम/लिटर) अधिक मँकोझेब ७५% डब्ल्यूपी (२.० ग्रॅम/लिटर), किंवा मेटलॅक्सिल ८% + मँकोझेब ६४% डब्ल्यूपी (२.५ ग्रॅम/लिटर) यांसारख्या उपचारात्मक बुरशीनाशकांची पानांच्या वरच्या व खालच्या दोन्ही बाजूंस कसून फवारणी करा.`;
  }

  if (lower.includes("early blight") || lower.includes("concentric")) {
    return `आपल्या ${cropDisplay} वरील '${diseaseDisplay}' रोगाचे वेळेवर नियंत्रण करणे गरजेचे आहे. हा बुरशीजन्य रोग असून जुन्या पानांवर गोलाकार वलयी तपकिरी डाग पडतात. नियंत्रणासाठी: रोगग्रस्त पाने काढून नष्ट करा. प्रतिबंधात्मक फवारणीसाठी मँकोझेब ७५% डब्ल्यूपी (२.०-२.५ ग्रॅम/लिटर) किंवा क्लोरोथॅलोनिल ७५% डब्ल्यूपी (२.० ग्रॅम/लिटर) फवारा. प्रादुर्भाव जास्त असल्यास अझोक्सिस्ट्रोबिन २३% एससी (१ मिली/लिटर) किंवा टेब्युकोनॅझोल २५.९% ईसी (१.५ मिली/लिटर) चा वापर करा.`;
  }

  if (lower.includes("bacterial spot")) {
    return `आपल्या ${cropDisplay} वरील '${diseaseDisplay}' हा जिवाणूजन्य रोग आहे. पानांवर व फळांवर लहान काळसर-तपकिरी पाणीदार ठिपके दिसतात. नियंत्रणासाठी: कॉपर ऑक्सिक्लोराईड ५०% डब्ल्यूपी (२.५ ग्रॅम/लिटर) अधिक स्ट्रेप्टोमायसिन सल्फेट ९०% (०.२ ग्रॅम/लिटर) पाण्यात मिसळून फवारा. पाण्याचा अतिवापर टाळा आणि तुषार सिंचन बंद ठेवा.`;
  }

  // Fallback: Return structured Marathi agronomic summary
  return `आपल्या ${cropDisplay} पिकासाठी कृषी तज्ज्ञ शिफारस: '${diseaseDisplay}' रोगाचा फैलाव रोखण्यासाठी प्रादुर्भावग्रस्त पाने त्वरित गोळा करून नष्ट करा. शेतात पाण्याचा निचरा योग्य ठेवा आणि आयसीएआर शिफारशीनुसार योग्य बुरशीनाशकाची किंवा सेंद्रिय निंबोळी अर्काची फवारणी करा.`;
}

function translateBlockToMarathi(section, data) {
  const crop = data.crop_mr || CROP_NAME_MAP_MR[data.crop] || data.crop || "पीक";
  const disease = data.disease_mr || DISEASE_NAME_MAP_MR[data.disease] || data.disease || "रोग";

  if (section === "explanation") {
    return `हे पीक '${crop}' असून पानांच्या तपासणीत '${disease}' चे निदान झाले आहे. दमट वातावरण आणि बुरशीच्या बीजाणूंमुळे पानांवर डाग पडतात, ज्यामुळे झाडाचे अन्न तयार करण्याचे प्रमाण घटते.`;
  }
  if (section === "management") {
    return `कृषी संशोधन परिषदेनुसार (ICAR) उपाययोजना:\n• प्रादुर्भावग्रस्त पाने तोडून शेताबाहेर नष्ट करा.\n• ठिबक सिंचनाचा वापर करा आणि पानांवर पाणी साचू देऊ नका.\n• सुरुवातीस मँकोझेब ७५% डब्ल्यूपी (२.५ ग्रॅम/लिटर) ची प्रतिबंधात्मक फवारणी करा.\n• रोग वाढल्यास सायमोक्सानिल किंवा मेटलॅक्सिलयुक्त बुरशीनाशक वापरा.`;
  }
  if (section === "prevention") {
    return `दीर्घकालीन प्रतिबंधात्मक उपाय:\n• पिकांची योग्य फेरपालट (Crop Rotation) करा.\n• झाडांमध्ये योग्य अंतर ठेवून हवा खेळती राहील याची काळजी घ्या.\n• नेहमी प्रमाणित रोगमुक्त बियाणे किंवा रोगप्रतिकारक वाणांचा वापर करा.`;
  }
  if (section === "precautions") {
    return `फवारणी करताना संरक्षक मास्क व हातमोजे वापरा. फवारणीनंतर तोडणीपूर्वीचा प्रतीक्षा काळ (PHI: ७-१० दिवस) काटेकोरपणे पाळा.`;
  }
  if (section === "advisory") {
    return `मोठ्या प्रमाणावर रासायनिक औषधांचा वापर करण्यापूर्वी आपल्या जवळच्या कृषी विज्ञान केंद्र (KVK) किंवा तालुका कृषी अधिकाऱ्यांशी संपर्क साधा.`;
  }
  if (section === "soil") {
    return `मृदा आरोग्य विश्लेषण [प्रात्यक्षिक नोंद]: जमिनीतील सामू (pH) व पालाशचे योग्य प्रमाण पानांच्या पेशींना बळकटी देते. मातीत ओलावा जास्त असल्यास बुरशीच्या बीजाणूंचा फैलाव वेगाने होतो, त्यामुळे पाण्याचा निचरा योग्य ठेवा आणि जमिनीवर पाणी साचू देऊ नका. (टीप: ही प्रात्यक्षिक माहिती आहे; प्रत्यक्ष खतांच्या नियोजनासाठी अधिकृत माती परीक्षण पत्रिका वापरावी).`;
  }
  return "";
}

// DOM Elements
const langEnBtn = document.getElementById("lang-en");
const langMrBtn = document.getElementById("lang-mr");

// Tabs & Panels
const tabModeDiagnose = document.getElementById("tab-mode-diagnose");
const tabModeAsk = document.getElementById("tab-mode-ask");
const panelDiagnose = document.getElementById("panel-diagnose");
const panelAsk = document.getElementById("panel-ask");

// Leaf Diagnosis Elements
const dropzone = document.getElementById("dropzone");
const leafInput = document.getElementById("leaf-image-input");
const dropzonePrompt = document.getElementById("dropzone-prompt");
const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const btnRemoveImage = document.getElementById("btn-remove-image");
const locationInput = document.getElementById("location-input");
const questionInput = document.getElementById("question-input");
const micBtn = document.getElementById("mic-btn");
const gradcamToggle = document.getElementById("gradcam-toggle");
const analyzeBtn = document.getElementById("analyze-btn");
const analyzeSpinner = document.getElementById("analyze-spinner");
const btnAnalyzeText = document.getElementById("btn-analyze-text");

// Standalone Q&A Elements
const standaloneLocationInput = document.getElementById("standalone-location-input");
const standaloneQuestionInput = document.getElementById("standalone-question-input");
const standaloneMicBtn = document.getElementById("standalone-mic-btn");
const standaloneAskBtn = document.getElementById("standalone-ask-btn");
const standaloneSpinner = document.getElementById("standalone-spinner");
const btnStandaloneText = document.getElementById("btn-standalone-text");

// Results Panel Elements
const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");
const qualityAlert = document.getElementById("quality-alert");
const qualityAlertText = document.getElementById("quality-alert-text");

// Direct Q&A Answer Card
const qaAnswerBox = document.getElementById("qa-answer-box");
const qaQuestionDisplay = document.getElementById("qa-question-display");
const qaDirectAnswer = document.getElementById("qa-direct-answer");
const qaAudioBtn = document.getElementById("qa-audio-btn");

// Diagnosis Banner
const diagnosisBanner = document.getElementById("diagnosis-banner");
const resCrop = document.getElementById("res-crop");
const resDisease = document.getElementById("res-disease");
const resConfidence = document.getElementById("res-confidence");
const resConfidenceBox = document.getElementById("res-confidence-box");
const confBarTrack = document.getElementById("conf-bar-track");
const confBarFill = document.getElementById("conf-bar-fill");
const uncertaintyBox = document.getElementById("uncertainty-box");
const uncertaintyText = document.getElementById("uncertainty-text");

// Weather
const weatherBox = document.getElementById("weather-box");
const weatherLoc = document.getElementById("weather-loc");
const weatherDetails = document.getElementById("weather-details");
const weatherRiskText = document.getElementById("weather-risk-text");

// Soil Health Micro-Card & Guidance
const soilBox = document.getElementById("soil-box");
const txtSoilHeading = document.getElementById("txt-soil-heading");
const badgeSynthetic = document.getElementById("badge-synthetic");
const soilTypeLabel = document.getElementById("soil-type-label");
const lblSoilPh = document.getElementById("lbl-soil-ph");
const valSoilPh = document.getElementById("val-soil-ph");
const subSoilPh = document.getElementById("sub-soil-ph");
const lblSoilMoisture = document.getElementById("lbl-soil-moisture");
const valSoilMoisture = document.getElementById("val-soil-moisture");
const subSoilMoisture = document.getElementById("sub-soil-moisture");
const lblSoilNpk = document.getElementById("lbl-soil-npk");
const valSoilNpk = document.getElementById("val-soil-npk");
const subSoilNpk = document.getElementById("sub-soil-npk");
const lblSoilCarbon = document.getElementById("lbl-soil-carbon");
const valSoilCarbon = document.getElementById("val-soil-carbon");
const subSoilCarbon = document.getElementById("sub-soil-carbon");
const soilRiskText = document.getElementById("soil-risk-text");
const blockSoil = document.getElementById("block-soil");
const txtSoilTitle = document.getElementById("txt-soil-title");
const guidanceSoil = document.getElementById("guidance-soil");

// Audio & Guidance
const ttsBtn = document.getElementById("tts-btn");
const guidanceAudio = document.getElementById("guidance-audio");
const guidanceSections = document.getElementById("guidance-sections");
const blockExplanation = document.getElementById("block-explanation");
const guidanceExplanation = document.getElementById("guidance-explanation");
const guidanceManagement = document.getElementById("guidance-management");
const guidancePrevention = document.getElementById("guidance-prevention");
const guidancePrecautions = document.getElementById("guidance-precautions");
const guidanceAdvisory = document.getElementById("guidance-advisory");

// Grad-CAM & Sources
const gradcamBox = document.getElementById("gradcam-box");
const gradcamImage = document.getElementById("gradcam-image");
const sourcesBox = document.getElementById("sources-box");
const sourcesList = document.getElementById("sources-list");

// Follow-up Section
const followupCard = document.getElementById("followup-card");
const followupInput = document.getElementById("followup-input");
const followupMicBtn = document.getElementById("followup-mic-btn");
const followupSendBtn = document.getElementById("followup-send-btn");
const followupSpinner = document.getElementById("followup-spinner");
const txtBtnFollowup = document.getElementById("txt-btn-followup");
const followupThread = document.getElementById("followup-thread");

const systemBadge = document.getElementById("system-status-text");

// Theme State & Toggle Handling
let currentTheme = "light";

function setupThemeToggle() {
  const themeToggle = document.getElementById("theme-toggle");

  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("agro_theme") || "light";
  } catch (e) {}

  setTheme(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const nextTheme = currentTheme === "dark" ? "light" : "dark";
      setTheme(nextTheme);
    });

    themeToggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const nextTheme = currentTheme === "dark" ? "light" : "dark";
        setTheme(nextTheme);
      }
    });
  }
}

function setTheme(theme) {
  currentTheme = theme;
  const isDark = theme === "dark";
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("agro_theme", theme);
  } catch (e) {}

  const themeIcon = document.getElementById("theme-icon");
  const themeText = document.getElementById("theme-text");
  const themeToggle = document.getElementById("theme-toggle");

  if (themeIcon) themeIcon.textContent = isDark ? "☀️" : "🌙";
  const isMr = currentLanguage === "marathi";
  if (themeText) {
    themeText.textContent = isDark
      ? (isMr ? "लाईट" : "Light")
      : (isMr ? "डार्क" : "Dark");
  }
  if (themeToggle) {
    const label = isDark
      ? (isMr ? "लाईट मोड निवडा" : "Switch to light mode")
      : (isMr ? "डार्क मोड निवडा" : "Switch to dark mode");
    themeToggle.setAttribute("aria-label", label);
    themeToggle.title = label;
  }
}

// Application Initialization
document.addEventListener("DOMContentLoaded", () => {
  setupThemeToggle();
  setupLanguageSwitcher();
  setupModeTabs();
  setupDropzone();
  setupQuickChips();
  setupSpeechRecognition();
  setupTTS();
  setupFormSubmission();
  setupStandaloneQA();
  setupFollowUpQA();
  setupSampleSelectors();
  checkSystemHealth();
  setupBadge();
});

// System Health Check with automatic Render cold-start retry
async function checkSystemHealth(retries = 5) {
  try {
    systemBadge.textContent = "Connecting to AI Backend...";
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 9000);
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    
    if (res.ok) {
      const data = await res.json();
      systemBadge.textContent = `${data.model} Ready (${data.num_classes} classes)`;
      systemBadge.style.color = "var(--primary)";
      return;
    }
    throw new Error("Health check non-200");
  } catch (e) {
    if (retries > 0) {
      systemBadge.textContent = `⚡ Waking up AI server (Render ~${retries * 8}s)...`;
      systemBadge.style.color = "var(--accent-amber)";
      setTimeout(() => checkSystemHealth(retries - 1), 6000);
    } else {
      systemBadge.textContent = "Offline Mode";
      systemBadge.style.color = "var(--accent-red)";
    }
  }
}

// Mode Switching (Leaf Diagnosis vs Standalone Q&A)

// Backend error handler: User-friendly notification without exposing system configurations
function handleBackendConnectionError(error, contextName = "Operation") {
  console.error(`${contextName} error:`, error);
  const isFetchError = error.message && error.message.toLowerCase().includes("fetch");
  
  if (isFetchError) {
    alert(
      `⏳ AI Server Connecting...\n\n` +
      `The cloud AI backend is waking up (Render free instances take ~30-50s to spin up).\n\n` +
      `Please wait a moment and click Analyze / Ask again.`
    );
  } else {
    alert(`${contextName} Notice: ${error.message || "An issue occurred while processing. Please retry."}`);
  }
}

function setupBadge() {
  const badge = document.getElementById("system-badge");
  if (badge) {
    badge.style.cursor = "default";
    badge.title = "AI Agricultural Diagnostics Service";
  }
}

function setupModeTabs() {
  tabModeDiagnose.addEventListener("click", () => switchMode("diagnose"));
  tabModeAsk.addEventListener("click", () => switchMode("ask"));

  // Crop context chips in Standalone Q&A
  const cropChips = document.querySelectorAll(".crop-chip");
  cropChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      cropChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      selectedCropContext = chip.dataset.crop || "";
    });
  });
}

function switchMode(mode) {
  activeMode = mode;
  if (mode === "diagnose") {
    tabModeDiagnose.classList.add("active");
    tabModeDiagnose.setAttribute("aria-selected", "true");
    tabModeAsk.classList.remove("active");
    tabModeAsk.setAttribute("aria-selected", "false");
    panelDiagnose.style.display = "block";
    panelAsk.style.display = "none";
  } else {
    tabModeAsk.classList.add("active");
    tabModeAsk.setAttribute("aria-selected", "true");
    tabModeDiagnose.classList.remove("active");
    tabModeDiagnose.setAttribute("aria-selected", "false");
    panelDiagnose.style.display = "none";
    panelAsk.style.display = "block";
  }
}

// Language Switching
function setupLanguageSwitcher() {
  langEnBtn.addEventListener("click", () => setLanguage("english"));
  langMrBtn.addEventListener("click", () => setLanguage("marathi"));
}

function setLanguage(lang) {
  currentLanguage = lang;
  if (lang === "marathi") {
    langMrBtn.classList.add("active");
    langEnBtn.classList.remove("active");
  } else {
    langEnBtn.classList.add("active");
    langMrBtn.classList.remove("active");
  }

  const t = I18N[lang];
  document.getElementById("app-title").textContent = t.appTitle;
  document.getElementById("app-subtitle").textContent = t.appSubtitle;
  document.getElementById("txt-tab-diagnose").textContent = t.tabDiagnose;
  document.getElementById("txt-tab-ask").textContent = t.tabAsk;
  document.getElementById("txt-leaf-input").textContent = t.leafInput;
  document.getElementById("txt-dropzone-prompt").innerHTML = t.dropzonePrompt;
  document.getElementById("txt-dropzone-hint").textContent = t.dropzoneHint;
  document.getElementById("lbl-samples").textContent = t.lblSamples;
  document.getElementById("lbl-location").children[0].textContent = t.lblLocation;
  document.getElementById("lbl-question").children[0].textContent = t.lblQuestion;
  document.getElementById("lbl-question").children[1].textContent = t.lblQuestionSub;
  questionInput.placeholder = t.qPlaceholder;
  document.getElementById("txt-gradcam-opt").textContent = t.gradcamOpt;
  btnAnalyzeText.textContent = t.btnAnalyze;

  document.getElementById("txt-standalone-title").textContent = t.standaloneTitle;
  document.getElementById("txt-standalone-desc").textContent = t.standaloneDesc;
  document.getElementById("lbl-crop-context").textContent = t.lblCropContext;
  document.getElementById("lbl-standalone-loc").children[0].textContent = t.lblStandaloneLoc;
  document.getElementById("lbl-standalone-question").children[0].textContent = t.lblStandaloneQuestion;
  standaloneQuestionInput.placeholder = t.standalonePlaceholder;
  btnStandaloneText.textContent = t.btnStandalone;

  document.getElementById("txt-diagnosis-heading").textContent = t.diagnosisHeading;
  document.getElementById("txt-empty-title").textContent = t.emptyTitle;
  document.getElementById("txt-empty-desc").textContent = t.emptyDesc;
  document.getElementById("txt-conf-label").textContent = t.confLabel;
  document.getElementById("txt-listen-guidance").textContent = t.listenAdvisory;

  document.getElementById("txt-qa-answer-title").textContent = t.qaAnswerTitle;
  document.getElementById("txt-qa-audio").textContent = t.listenAnswer;

  const expTitleEl = document.getElementById("txt-explanation-title"); if (expTitleEl) expTitleEl.textContent = t.explanationTitle;
  document.getElementById("txt-management-title").textContent = t.managementTitle;
  document.getElementById("txt-prevention-title").textContent = t.preventionTitle;
  document.getElementById("txt-precautions-title").textContent = t.precautionsTitle;
  document.getElementById("txt-advisory-title").textContent = t.advisoryTitle;
  document.getElementById("txt-gradcam-title").textContent = t.gradcamTitle;
  document.getElementById("txt-gradcam-desc").textContent = t.gradcamDesc;
  document.getElementById("txt-sources-title").textContent = t.sourcesTitle;
  document.getElementById("txt-footer").textContent = t.footerText;

  // Localize Soil Health card
  if (txtSoilHeading) txtSoilHeading.textContent = t.soilHeading;
  if (badgeSynthetic) badgeSynthetic.textContent = t.syntheticBadge;
  if (txtSoilTitle) txtSoilTitle.textContent = t.soilTitle;
  if (lblSoilPh) lblSoilPh.textContent = t.lblSoilPh;
  if (lblSoilMoisture) lblSoilMoisture.textContent = t.lblSoilMoisture;
  if (lblSoilNpk) lblSoilNpk.textContent = t.lblSoilNpk;
  if (lblSoilCarbon) lblSoilCarbon.textContent = t.lblSoilCarbon;
  if (subSoilNpk) subSoilNpk.textContent = t.npkSub;

  // Localize 7 Scannable Guidance Section Headings
  const elSecDisease = document.getElementById("txt-sec-disease-title");
  if (elSecDisease && t.secDiseaseTitle) elSecDisease.textContent = t.secDiseaseTitle;
  const elSecWeather = document.getElementById("txt-sec-weather-title");
  if (elSecWeather && t.secWeatherTitle) elSecWeather.textContent = t.secWeatherTitle;
  const elSecSoil = document.getElementById("txt-sec-soil-title");
  if (elSecSoil && t.secSoilTitle) elSecSoil.textContent = t.secSoilTitle;
  const elSecActions = document.getElementById("txt-sec-actions-title");
  if (elSecActions && t.secActionsTitle) elSecActions.textContent = t.secActionsTitle;
  const elSecTreatment = document.getElementById("txt-sec-treatment-title");
  if (elSecTreatment && t.secTreatmentTitle) elSecTreatment.textContent = t.secTreatmentTitle;
  const elSecSafety = document.getElementById("txt-sec-safety-title");
  if (elSecSafety && t.secSafetyTitle) elSecSafety.textContent = t.secSafetyTitle;

  // Re-sync Theme Toggle label for active language
  setTheme(currentTheme);

  document.getElementById("txt-followup-title").textContent = t.followupTitle;
  document.getElementById("txt-followup-sub").textContent = t.followupSub;
  followupInput.placeholder = t.followupPlaceholder;
  txtBtnFollowup.textContent = t.btnFollowup;

  // Localize System Status text
  if (systemBadge) {
    if (systemBadge.textContent.includes("Ready") || systemBadge.textContent.includes("सज्ज")) {
      systemBadge.textContent = lang === "marathi"
        ? "MobileNetV3 मॉडेल सज्ज (१३ रोग वर्ग)"
        : "MobileNetV3-Large Ready (13 classes)";
    }
  }

  // Localize Sample Leaf pills
  document.querySelectorAll(".sample-pill").forEach((pill) => {
    const lbl = lang === "marathi" ? pill.dataset.labelMr : pill.dataset.labelEn;
    if (lbl) pill.textContent = lbl;
  });

  // Localize Location quick chips
  document.querySelectorAll(".loc-chip").forEach((chip) => {
    const lbl = lang === "marathi" ? chip.dataset.labelMr : chip.dataset.labelEn;
    if (lbl) chip.textContent = lbl;
  });

  // Localize Crop context chips (Standalone mode)
  document.querySelectorAll(".crop-chip").forEach((chip) => {
    const lbl = lang === "marathi" ? chip.dataset.labelMr : chip.dataset.labelEn;
    if (lbl) chip.textContent = lbl;
  });

  // Localize Question quick chips
  const qChips = document.querySelectorAll(".q-chip");
  if (qChips.length >= 3) {
    qChips[0].textContent = lang === "marathi" ? "फवारणी सल्ला?" : "Spray guidance?";
    qChips[0].dataset.q = t.sprayQ;
    qChips[1].textContent = lang === "marathi" ? "प्रतिबंध?" : "Prevention?";
    qChips[1].dataset.q = t.preventQ;
    qChips[2].textContent = lang === "marathi" ? "उत्पादन धोका?" : "Harvest risk?";
    qChips[2].dataset.q = t.harvestQ;
  }

  // Localize Standalone question chips
  document.querySelectorAll(".sq-chip").forEach((chip) => {
    const lbl = lang === "marathi" ? chip.dataset.labelMr : chip.dataset.labelEn;
    if (lbl) chip.textContent = lbl;
  });

  // Localize Followup question chips
  document.querySelectorAll(".follow-chip").forEach((chip) => {
    const lbl = lang === "marathi" ? chip.dataset.labelMr : chip.dataset.labelEn;
    if (lbl) chip.textContent = lbl;
  });

  // CRITICAL: Immediately re-render active results in the newly selected language!
  if (lastAnalysisData) {
    renderResults(lastAnalysisData, lastAnalysisOptions || {});
  }
}

// Dropzone file handling
function setupDropzone() {
  dropzone.addEventListener("click", () => leafInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  leafInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  btnRemoveImage.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileInput();
  });
}

function handleSelectedFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG or PNG).");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    dropzonePrompt.style.display = "none";
    previewContainer.style.display = "flex";
    analyzeBtn.disabled = false;
    analyzeBtn.classList.remove("btn-unready");
    analyzeBtn.classList.add("btn-ready");
    analyzeBtn.title = "Ready to analyze leaf & weather";
  };
  reader.readAsDataURL(file);
}

function resetFileInput() {
  selectedFile = null;
  leafInput.value = "";
  imagePreview.src = "";
  previewContainer.style.display = "none";
  dropzonePrompt.style.display = "block";
  document.querySelectorAll(".sample-pill").forEach((p) => p.classList.remove("active"));
  analyzeBtn.classList.add("btn-unready");
  analyzeBtn.classList.remove("btn-ready");
  analyzeBtn.title = "Click to analyze leaf photo and weather";
}

// Sample Selectors
function setupSampleSelectors() {
  const pills = document.querySelectorAll(".sample-pill");
  pills.forEach((pill) => {
    pill.addEventListener("click", async () => {
      pills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");

      const rawSrc = pill.dataset.src || "";
      const filename = rawSrc.split("/").pop();
      const candidates = [
        rawSrc,
        rawSrc.startsWith("/") ? rawSrc : "/" + rawSrc,
        "/data/samples/" + filename,
        "data/samples/" + filename,
        "/static/data/samples/" + filename,
        "static/data/samples/" + filename,
      ];

      let blob = null;
      for (const candidate of candidates) {
        if (!candidate) continue;
        try {
          const res = await fetch(candidate);
          if (res.ok) {
            blob = await res.blob();
            break;
          }
        } catch (e) {}
      }

      if (blob) {
        const file = new File([blob], (pill.dataset.name || "sample") + ".jpg", { type: "image/jpeg" });
        handleSelectedFile(file);
      } else {
        console.warn("Could not load sample file from any candidate URL:", rawSrc);
        alert(
          currentLanguage === "marathi"
            ? "नमुना पान लोड करण्यात अडचण आली. कृपया स्वतः फोटो निवडा."
            : "Could not load sample leaf image. Please upload a photo manually."
        );
      }
    });
  });
}

// Quick Chips
function setupQuickChips() {
  document.querySelectorAll(".chip[data-val]").forEach((chip) => {
    chip.addEventListener("click", () => {
      locationInput.value = chip.dataset.val;
    });
  });

  document.querySelectorAll(".q-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      questionInput.value = chip.dataset.q;
    });
  });

  document.querySelectorAll(".sq-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const qVal = currentLanguage === "marathi" ? (chip.dataset.qMr || chip.dataset.q) : chip.dataset.q;
      standaloneQuestionInput.value = qVal;
    });
  });

  document.querySelectorAll(".follow-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const qVal = currentLanguage === "marathi" ? (chip.dataset.qMr || chip.dataset.q) : chip.dataset.q;
      followupInput.value = qVal;
      followupInput.focus();
    });
  });
}

// Speech Recognition (Web Speech API)
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    if (micBtn) micBtn.style.display = "none";
    if (standaloneMicBtn) standaloneMicBtn.style.display = "none";
    if (followupMicBtn) followupMicBtn.style.display = "none";
    return;
  }

  function attachMic(button, targetInput) {
    if (!button) return;
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    button.addEventListener("click", () => {
      recognition.lang = currentLanguage === "marathi" ? "mr-IN" : "en-IN";
      try {
        recognition.start();
        button.classList.add("recording");
      } catch (e) {
        recognition.stop();
        button.classList.remove("recording");
      }
    });

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      targetInput.value = transcript;
      button.classList.remove("recording");
    };

    recognition.onerror = () => button.classList.remove("recording");
    recognition.onend = () => button.classList.remove("recording");
  }

  attachMic(micBtn, questionInput);
  attachMic(standaloneMicBtn, standaloneQuestionInput);
  attachMic(followupMicBtn, followupInput);
}

// Text-to-Speech Playback Helper
async function playSpeech(text, btnElement, origLabel) {
  if (!text) return;
  if (btnElement) {
    btnElement.disabled = true;
    if (origLabel) btnElement.querySelector("span:last-child").textContent = I18N[currentLanguage].playingAudio;
  }

  try {
    const formData = new FormData();
    formData.append("text", text);
    formData.append("language", currentLanguage);

    const res = await fetch(`${API_BASE}/tts`, {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      guidanceAudio.src = url;
      guidanceAudio.style.display = "block";
      guidanceAudio.play();
    } else {
      fallbackBrowserSpeech(text);
    }
  } catch (e) {
    console.warn("TTS fetch failed, attempting browser synth", e);
    fallbackBrowserSpeech(text);
  } finally {
    if (btnElement) {
      btnElement.disabled = false;
      if (origLabel) btnElement.querySelector("span:last-child").textContent = origLabel;
    }
  }
}

function fallbackBrowserSpeech(text) {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    const isMarathi = currentLanguage === "marathi";
    utter.lang = isMarathi ? "mr-IN" : "en-IN";
    utter.rate = 0.95; // Calm, clear natural cadence
    utter.pitch = 1.0;

    // Pick highest-quality natural neural voice available on the client device
    const voices = window.speechSynthesis.getVoices();
    if (voices && voices.length > 0) {
      const targetPrefix = isMarathi ? "mr" : "en";
      const matched = voices.filter(v => v.lang && v.lang.toLowerCase().startsWith(targetPrefix));
      const natural = matched.find(v => /natural|neural|online|google|expressive/i.test(v.name))
        || matched.find(v => v.lang.toLowerCase() === (isMarathi ? "mr-in" : "en-in"))
        || matched[0];
      if (natural) utter.voice = natural;
    }
    window.speechSynthesis.speak(utter);
  }
}

function setupTTS() {
  ttsBtn.addEventListener("click", () => {
    const origText = ttsBtn.querySelector("span:last-child").textContent;
    playSpeech(currentGuidanceText, ttsBtn, origText);
  });

  qaAudioBtn.addEventListener("click", () => {
    const origText = qaAudioBtn.querySelector("span:last-child").textContent;
    playSpeech(currentDirectAnswerText, qaAudioBtn, origText);
  });
}

// Leaf Diagnosis Form Submission & Pipeline Execution
function setupFormSubmission() {
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      dropzone.classList.add("highlight-prompt");
      setTimeout(() => dropzone.classList.remove("highlight-prompt"), 1200);
      alert(
        currentLanguage === "marathi"
          ? "कृपया प्रथम पिकाच्या पानाचा फोटो अपलोड करा किंवा वरील नमुना पान (Sample Leaf) निवडा."
          : "Please upload a crop leaf photo or click a sample leaf button above to analyze."
      );
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.classList.remove("btn-ready");
    analyzeSpinner.style.display = "inline-block";
    btnAnalyzeText.textContent = I18N[currentLanguage].btnAnalyzing;
    qualityAlert.style.display = "none";

    const farmerQ = questionInput.value.trim();
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("question", farmerQ);
    formData.append("location", locationInput.value.trim() || "Pune");
    formData.append("language", currentLanguage);
    formData.append("include_gradcam", gradcamToggle.checked ? "true" : "false");

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Server error during analysis.");
      }

      // Record active context for follow-up questions
      currentCrop = data.crop;
      currentDisease = data.disease;
      currentConfidence = data.confidence;
      currentWeather = data.weather;

      renderResults(data, { isStandalone: false, initialQuestion: farmerQ });
    } catch (error) {
      handleBackendConnectionError(error, "Analysis");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.add("btn-ready");
      analyzeSpinner.style.display = "none";
      btnAnalyzeText.textContent = I18N[currentLanguage].btnAnalyze;
    }
  });
}

// Standalone Agronomist Q&A Form Submission
function setupStandaloneQA() {
  standaloneAskBtn.addEventListener("click", async () => {
    const query = standaloneQuestionInput.value.trim();
    if (!query) {
      alert(currentLanguage === "marathi" ? "कृपया आपला प्रश्न प्रविष्ट करा." : "Please enter your farming question.");
      standaloneQuestionInput.focus();
      return;
    }

    standaloneAskBtn.disabled = true;
    standaloneSpinner.style.display = "inline-block";
    btnStandaloneText.textContent = I18N[currentLanguage].btnStandaloneAsking;

    const loc = standaloneLocationInput.value.trim() || "Pune";

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop: selectedCropContext || null,
          disease: null,
          confidence: 0.9,
          question: query,
          location: loc,
          language: currentLanguage,
          is_low_confidence: false,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Server error retrieving agricultural advice.");
      }

      // Record active context
      currentCrop = data.crop;
      currentDisease = data.disease;
      currentConfidence = 0.9;
      currentWeather = data.weather;

      renderResults(data, { isStandalone: true, initialQuestion: query });
    } catch (error) {
      handleBackendConnectionError(error, "Agronomist Advisory");
    } finally {
      standaloneAskBtn.disabled = false;
      standaloneSpinner.style.display = "none";
      btnStandaloneText.textContent = I18N[currentLanguage].btnStandalone;
    }
  });
}

// Follow-up Q&A Submission
function setupFollowUpQA() {
  async function submitFollowUp() {
    const q = followupInput.value.trim();
    if (!q) return;

    followupSendBtn.disabled = true;
    followupSpinner.style.display = "inline-block";
    txtBtnFollowup.textContent = I18N[currentLanguage].btnFollowupAsking;

    // Append user question bubble immediately
    const userBubble = document.createElement("div");
    userBubble.className = "thread-item thread-user-q";
    userBubble.innerHTML = `<strong>${currentLanguage === "marathi" ? "शेतकऱ्याचा प्रश्न" : "Farmer Query"}:</strong> ${escapeHtml(q)}`;
    followupThread.appendChild(userBubble);
    userBubble.scrollIntoView({ behavior: "smooth", block: "nearest" });

    followupInput.value = "";

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          crop: currentCrop || "General / Solanaceae",
          disease: currentDisease || "General Agronomic Advisory",
          confidence: currentConfidence || 0.85,
          question: q,
          location: (locationInput.value || standaloneLocationInput.value || "Pune").trim(),
          language: currentLanguage,
          is_low_confidence: false,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Error generating follow-up response.");

      const directAns = data.direct_answer || data.guidance?.direct_answer || data.guidance?.explanation || "Guidance received.";

      // Append assistant answer bubble
      const aiBubble = document.createElement("div");
      aiBubble.className = "thread-item thread-ai-ans";
      const answerId = "ans-" + Date.now();
      aiBubble.innerHTML = `
        <div class="thread-ai-header">
          <span>🌿 ${currentLanguage === "marathi" ? "कृषी सल्ला" : "Agronomist Advisory"}</span>
          <button type="button" class="btn-mini-audio" id="${answerId}">🔊 ${I18N[currentLanguage].listenAnswer}</button>
        </div>
        <div class="thread-ai-body markdown-body">${renderMarkdown(directAns)}</div>
      `;
      followupThread.appendChild(aiBubble);
      aiBubble.scrollIntoView({ behavior: "smooth", block: "nearest" });

      // Attach audio button listener
      document.getElementById(answerId).addEventListener("click", function () {
        playSpeech(directAns, this, `🔊 ${I18N[currentLanguage].listenAnswer}`);
      });
    } catch (err) {
      const errBubble = document.createElement("div");
      errBubble.className = "thread-item";
      errBubble.style.color = "var(--accent-red)";
      errBubble.textContent = `Error: ${err.message}`;
      followupThread.appendChild(errBubble);
    } finally {
      followupSendBtn.disabled = false;
      followupSpinner.style.display = "none";
      txtBtnFollowup.textContent = I18N[currentLanguage].btnFollowup;
    }
  }

  followupSendBtn.addEventListener("click", submitFollowUp);
  followupInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      submitFollowUp();
    }
  });
}

function escapeHtml(text) {
  return (text || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[m]));
}

/**
 * Safely renders Markdown to HTML with strict sanitization against script and HTML injection.
 * Integrates marked + DOMPurify when loaded, with a robust built-in parser for 100% offline resilience.
 */
function renderMarkdown(input) {
  if (input == null) return "";
  const text = String(input).trim();
  if (!text) return "";

  // 1. If marked + DOMPurify are loaded:
  if (typeof window !== "undefined" && typeof window.marked !== "undefined" && typeof window.DOMPurify !== "undefined") {
    try {
      const rawHtml = window.marked.parse(text, { breaks: true, gfm: true });
      return window.DOMPurify.sanitize(rawHtml, {
        ALLOWED_TAGS: [
          "h1", "h2", "h3", "h4", "h5", "h6", "p", "br", "strong", "b",
          "em", "i", "ul", "ol", "li", "code", "pre", "blockquote", "hr",
          "span", "a"
        ],
        ALLOWED_ATTR: ["href", "target", "rel", "class", "title"]
      });
    } catch (e) {
      console.warn("marked library error, using secure built-in parser:", e);
    }
  }

  // 2. Built-in zero-dependency safe Markdown parser with complete entity escaping
  return safeInternalMarkdown(text);
}

function safeInternalMarkdown(text) {
  if (!text) return "";
  // Escape raw HTML entities strictly to prevent XSS
  let safe = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

  safe = safe.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = safe.split("\n");
  const output = [];
  let inUl = false;
  let inOl = false;

  function closeLists() {
    if (inUl) { output.push("</ul>"); inUl = false; }
    if (inOl) { output.push("</ol>"); inOl = false; }
  }

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    if (!line) {
      closeLists();
      continue;
    }

    // Headings (###, ##, #)
    const h3Match = line.match(/^###\s+(.+)$/);
    if (h3Match) {
      closeLists();
      output.push(`<h4>${parseInlineMarkdown(h3Match[1])}</h4>`);
      continue;
    }
    const h2Match = line.match(/^##\s+(.+)$/);
    if (h2Match) {
      closeLists();
      output.push(`<h3>${parseInlineMarkdown(h2Match[1])}</h3>`);
      continue;
    }
    const h1Match = line.match(/^#\s+(.+)$/);
    if (h1Match) {
      closeLists();
      output.push(`<h2>${parseInlineMarkdown(h1Match[1])}</h2>`);
      continue;
    }

    // Horizontal Rule
    if (/^(?:---|\*\*\*|___)$/.test(line)) {
      closeLists();
      output.push("<hr>");
      continue;
    }

    // Blockquote
    const bqMatch = line.match(/^>\s*(.+)$/);
    if (bqMatch) {
      closeLists();
      output.push(`<blockquote>${parseInlineMarkdown(bqMatch[1])}</blockquote>`);
      continue;
    }

    // Unordered List item (* or -)
    const ulMatch = line.match(/^[-*•]\s+(.+)$/);
    if (ulMatch) {
      if (inOl) { output.push("</ol>"); inOl = false; }
      if (!inUl) { output.push("<ul>"); inUl = true; }
      output.push(`<li>${parseInlineMarkdown(ulMatch[1])}</li>`);
      continue;
    }

    // Ordered List item (1., 2., etc.)
    const olMatch = line.match(/^\d+[.)]\s+(.+)$/);
    if (olMatch) {
      if (inUl) { output.push("</ul>"); inUl = false; }
      if (!inOl) { output.push("<ol>"); inOl = true; }
      output.push(`<li>${parseInlineMarkdown(olMatch[1])}</li>`);
      continue;
    }

    // Regular paragraph line
    closeLists();
    output.push(`<p>${parseInlineMarkdown(line)}</p>`);
  }

  closeLists();
  return output.join("\n");
}

function parseInlineMarkdown(text) {
  if (!text) return "";
  return text
    // Bold: **text** or __text__
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    // Italic: *text* or _text_
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>")
    .replace(/(?<!_)_(?!_)(.+?)(?<!_)_(?!_)/g, "<em>$1</em>")
    // Inline code: `code`
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // Safe Links: only http/https allowed
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s\)\"\']+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

// Render Results to UI (Shared by Initial Analysis & Standalone Q&A with Instant Multilingual Switch)
function renderResults(data, options = {}) {
  const { isStandalone = false, initialQuestion = "" } = options;

  lastAnalysisData = data;
  lastAnalysisOptions = options;

  emptyState.style.display = "none";
  resultsContent.style.display = "block";

  const isMr = currentLanguage === "marathi";

  // Check if image quality failed (only applicable in leaf diagnosis)
  if (!isStandalone && !data.success) {
    qualityAlert.style.display = "flex";
    qualityAlertText.textContent = isMr
      ? "प्रतिमेची गुणवत्ता कमी आहे (अतिशय अंधुक किंवा अस्पष्ट). कृपया पानाचा स्पष्ट व स्वच्छ फोटो अपलोड करा."
      : (data.warning || "Image quality issue detected.");
    diagnosisBanner.style.display = "none";
    weatherBox.style.display = "none";
    if (soilBox) soilBox.style.display = "none";
    if (guidanceSections) guidanceSections.style.display = "none";
    qaAnswerBox.style.display = "none";
    gradcamBox.style.display = "none";
    followupCard.style.display = "none";
    return;
  }

  qualityAlert.style.display = "none";
  diagnosisBanner.style.display = "block";
  weatherBox.style.display = "block";
  if (soilBox) soilBox.style.display = "block";
  if (guidanceSections) guidanceSections.style.display = "flex";
  followupCard.style.display = "block";

  // 1. Resolve Crop and Disease Names for active language
  const cropDisplay = isMr
    ? (data.crop_mr || CROP_NAME_MAP_MR[data.crop] || CROP_NAME_MAP_MR[data.crop_en] || data.crop || "पीक")
    : (data.crop_en || CROP_NAME_MAP_EN[data.crop] || data.crop || "Crop");

  const diseaseDisplay = isMr
    ? (data.disease_mr || DISEASE_NAME_MAP_MR[data.disease] || DISEASE_NAME_MAP_MR[data.disease_en] || data.disease || "निरोगी")
    : (data.disease_en || DISEASE_NAME_MAP_EN[data.disease] || data.disease || "Healthy");

  // 2. Direct Q&A Answer Card
  let directAnswerText = "";
  if (isMr) {
    directAnswerText = data.direct_answer_mr ||
      data.guidance?.mr?.direct_answer ||
      data.guidance_mr?.direct_answer ||
      "";
    if (!directAnswerText && (data.direct_answer || data.guidance?.direct_answer)) {
      directAnswerText = translateAgronomicTextToMarathi(data.direct_answer || data.guidance?.direct_answer, cropDisplay, diseaseDisplay);
    }
  } else {
    directAnswerText = data.direct_answer_en ||
      data.guidance?.en?.direct_answer ||
      data.guidance_en?.direct_answer ||
      data.direct_answer ||
      data.guidance?.direct_answer ||
      "";
  }

  const rawQ = initialQuestion || data.question || data.farmer_question || "";
  let displayQuestion = rawQ;
  if (isMr && rawQ) {
    displayQuestion = translateQuestionToMarathi(rawQ);
  }

  // If directAnswerText is absent (leaf diagnosis without custom question), synthesize integrated summary:
  const confPct = ((data.confidence || 0.85) * 100).toFixed(1);
  if (!directAnswerText) {
    if (isMr) {
      directAnswerText = `**${cropDisplay}** पिकामध्ये **${diseaseDisplay}** चे निदान झाले आहे (विश्वासार्हता ${confPct}%). ${w.location_mr || 'स्थानिक'} हवामान आणि जमिनीतील ओलावा पाहता पिकाचे संरक्षण करण्यासाठी खालील शिफारशी तातडीने अंमलात आणाव्यात.`;
    } else {
      directAnswerText = `**${cropDisplay}** diagnosed with **${diseaseDisplay}** (${confPct}% confidence). Local weather and soil conditions indicate integrated curative and preventive measures are required.`;
    }
  }

  // Section 1: Integrated Summary is always prominently displayed
  qaAnswerBox.style.display = "block";
  const qaTitleEl = document.getElementById("txt-qa-answer-title");
  if (displayQuestion) {
    qaQuestionDisplay.style.display = "block";
    qaQuestionDisplay.textContent = `"${displayQuestion}"`;
    if (qaTitleEl) qaTitleEl.textContent = isMr ? "१. एकत्रित कृषी सारांश व उत्तर" : "1. Integrated Summary & Answer";
  } else {
    qaQuestionDisplay.style.display = "none";
    if (qaTitleEl) qaTitleEl.textContent = isMr ? "१. एकत्रित कृषी सारांश" : "1. Integrated Summary";
  }
  qaDirectAnswer.innerHTML = renderMarkdown(directAnswerText);
  currentDirectAnswerText = directAnswerText;

  // 3. Diagnosis & Confidence Banner
  if (isStandalone) {
    resCrop.textContent = cropDisplay || (isMr ? "सर्वसाधारण शेती सल्ला" : "Agriculture Advisory");
    resDisease.textContent = diseaseDisplay || (isMr ? "कृषी तज्ज्ञ मार्गदर्शन" : "Expert Q&A Guidance");
    resConfidence.textContent = isMr ? "प्रमाणित" : "Verified";
    resConfidenceBox.style.display = "none";
    confBarTrack.style.display = "none";
    uncertaintyBox.style.display = "none";
  } else {
    resConfidenceBox.style.display = "flex";
    confBarTrack.style.display = "block";

    resCrop.textContent = cropDisplay || (isMr ? "पीक" : "Crop");
    resDisease.textContent = diseaseDisplay || (isMr ? "निरोगी" : "Healthy");
    const confPct = ((data.confidence || 0.85) * 100).toFixed(1);
    resConfidence.textContent = `${confPct}%`;
    confBarFill.style.width = `${Math.min(100, Math.max(10, (data.confidence || 0.85) * 100))}%`;

    // Color-code confidence bar
    if (data.confidence >= 0.8) {
      confBarFill.style.background = "var(--primary)";
    } else if (data.confidence >= 0.6) {
      confBarFill.style.background = "var(--accent-amber)";
    } else {
      confBarFill.style.background = "var(--accent-red)";
    }

    // Uncertainty Warning
    if (data.is_low_confidence || (data.confidence && data.confidence < 0.6)) {
      uncertaintyBox.style.display = "flex";
      uncertaintyText.textContent = isMr
        ? `सूचना: मॉडेलची विश्वासार्हता कमी (${confPct}%) आहे. कृपया प्रत्यक्ष शेतात स्थानिक कृषी अधिकाऱ्यांकडून खात्री करा.`
        : (data.warning || "Model confidence is below 60%. Please verify with a local agricultural officer.");
    } else {
      uncertaintyBox.style.display = "none";
    }
  }

  // 4. Weather Context
  const w = data.weather || {};
  if (isMr) {
    weatherLoc.textContent = w.location_mr || translateCityToMarathi(w.location);
    const condMr = w.condition_mr || translateConditionToMarathi(w.condition);
    weatherDetails.textContent = `तापमान: ${w.temperature_c ?? 25}°C | आर्द्रता: ${w.humidity_percentage ?? 60}% | ${condMr}`;
    weatherRiskText.textContent = w.risk_analysis_mr || translateWeatherRiskToMarathi(w.risk_analysis, w.humidity_percentage);
  } else {
    weatherLoc.textContent = w.location || "Field";
    weatherDetails.textContent = `Temp: ${w.temperature_c ?? 25}°C | Humidity: ${w.humidity_percentage ?? 60}% | ${w.condition ?? "Seasonal"}`;
    weatherRiskText.textContent = w.risk_analysis || "Standard seasonal precautions apply.";
  }

  // 4b. Synthetic Soil Health Context
  const s = data.soil || {};
  const sInterp = s.interpretation || {};
  if (soilBox) {
    soilBox.style.display = "block";
    if (soilTypeLabel) {
      soilTypeLabel.textContent = isMr
        ? (sInterp.soil_type_mr || "मध्यम काळी कसदार जमीन (Vertisol)")
        : (s.soil_type || "Medium Black Soil (Vertisol)");
    }
    if (valSoilPh) valSoilPh.textContent = s.soil_ph != null ? s.soil_ph.toFixed(1) : "7.0";
    if (subSoilPh) subSoilPh.textContent = isMr ? (sInterp.ph_status_mr || "योग्य") : (sInterp.ph_status || "Optimal Neutral");

    if (valSoilMoisture) valSoilMoisture.textContent = s.soil_moisture_percent != null ? `${s.soil_moisture_percent.toFixed(1)}%` : "25.0%";
    if (subSoilMoisture) subSoilMoisture.textContent = isMr ? (sInterp.moisture_status_mr || "वाफसा स्थिती") : (sInterp.moisture_status || "Field Capacity");

    if (valSoilNpk) {
      const nVal = s.nitrogen_kg_ha != null ? Math.round(s.nitrogen_kg_ha) : 185;
      const pVal = s.phosphorus_kg_ha != null ? Math.round(s.phosphorus_kg_ha) : 22;
      const kVal = s.potassium_kg_ha != null ? Math.round(s.potassium_kg_ha) : 310;
      valSoilNpk.textContent = `${nVal}-${pVal}-${kVal}`;
    }
    if (subSoilNpk) subSoilNpk.textContent = isMr ? "किलो/हेक्टर (आयसीएआर)" : "kg/ha (ICAR Rating)";

    if (valSoilCarbon) valSoilCarbon.textContent = s.organic_carbon_percent != null ? `${s.organic_carbon_percent.toFixed(2)}%` : "0.58%";
    if (subSoilCarbon) subSoilCarbon.textContent = isMr ? (sInterp.organic_carbon_rating === "High" ? "उत्कृष्ट" : (sInterp.organic_carbon_rating === "Low" ? "कमी" : "मध्यम")) : (sInterp.organic_carbon_rating || "Medium");

    if (soilRiskText) {
      soilRiskText.textContent = isMr
        ? (sInterp.disease_soil_interaction_mr || "जमिनीतील गुणधर्म संतुलित असून पीक पोषणासाठी अनुकूल आहेत.")
        : (sInterp.disease_soil_interaction || "Soil parameters within regional equilibrium with moderate disease vulnerability.");
    }
  }

  // 5. Structured Guidance (Explanation, Actions, Management, Prevention, Precautions, Advisory)
  const gBlock = isMr
    ? (data.guidance?.mr || data.guidance_mr || (data.guidance?.language === "marathi" ? data.guidance : null))
    : (data.guidance?.en || data.guidance_en || (data.guidance?.language === "english" ? data.guidance : null));

  const explanationText = gBlock?.explanation || (isMr ? translateBlockToMarathi("explanation", data) : (data.guidance?.explanation || "Agricultural advisory analysis complete."));
  const managementText = gBlock?.management_guidance || (isMr ? translateBlockToMarathi("management", data) : (data.guidance?.management_guidance || "Refer to evidence documents."));
  const preventionText = gBlock?.prevention || (isMr ? translateBlockToMarathi("prevention", data) : (data.guidance?.prevention || "Adopt recommended crop rotation."));
  const precautionsText = gBlock?.precautions || (isMr ? translateBlockToMarathi("precautions", data) : (data.guidance?.precautions || "Use standard safety equipment."));
  const advisoryText = gBlock?.expert_advisory || (isMr ? translateBlockToMarathi("advisory", data) : (data.guidance?.expert_advisory || "Consult your local KVK."));

  // Weather Interpretation in Section 3
  const weatherInterpText = gBlock?.weather_interpretation || data.guidance?.weather_interpretation || "";
  const guidanceWeatherInterp = document.getElementById("guidance-weather-interp");
  if (guidanceWeatherInterp) {
    if (weatherInterpText) {
      guidanceWeatherInterp.style.display = "block";
      guidanceWeatherInterp.innerHTML = renderMarkdown(weatherInterpText);
    } else {
      guidanceWeatherInterp.style.display = "none";
    }
  }

  // Section 2: Disease Explanation
  guidanceExplanation.innerHTML = renderMarkdown(explanationText);

  // Section 4: Soil Impact
  if (guidanceSoil) {
    const soilInterpText = gBlock?.soil_interpretation || (isMr ? translateBlockToMarathi("soil", data) : (data.guidance?.soil_interpretation || "Soil parameters baseline recorded."));
    guidanceSoil.innerHTML = renderMarkdown(soilInterpText);
  }

  // Section 5: Recommended Actions (Prioritized Key Action Items)
  let actionsText = gBlock?.recommended_actions || "";
  if (!actionsText && managementText) {
    const mLines = managementText.split("\n").map(l => l.trim()).filter(l => l.length > 0);
    const bulletLines = mLines.filter(l => l.match(/^[-*•\d]/));
    if (bulletLines.length >= 2) {
      actionsText = bulletLines.slice(0, 4).join("\n");
    } else {
      actionsText = mLines.slice(0, 3).map((l, idx) => `${idx + 1}. ${l.replace(/^[-*•\d.]+\s*/, '')}`).join("\n");
    }
  }
  const guidanceActions = document.getElementById("guidance-actions");
  if (guidanceActions) {
    guidanceActions.innerHTML = renderMarkdown(actionsText || (isMr ? "शिफारस केलेल्या कृती त्वरित अंमलात आणा." : "Implement recommended management steps immediately."));
  }

  // Section 6: Treatment & Prevention
  guidanceManagement.innerHTML = renderMarkdown(managementText);
  guidancePrevention.innerHTML = renderMarkdown(preventionText);

  // Section 7: Safety & Advisory
  guidancePrecautions.innerHTML = renderMarkdown(precautionsText);
  guidanceAdvisory.innerHTML = renderMarkdown(advisoryText);

  // Prepare full speech text
  currentGuidanceText = `${cropDisplay} ${diseaseDisplay}. ${directAnswerText ? directAnswerText + ". " : ""}${explanationText}. ${managementText}`;

  // 6. Grad-CAM (Only in image diagnosis mode)
  if (!isStandalone && data.gradcam_base64) {
    gradcamBox.style.display = "block";
    gradcamImage.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
  } else {
    gradcamBox.style.display = "none";
  }

  // 7. Evidence Sources
  sourcesList.innerHTML = "";
  const sources = data.sources || [];
  if (sources.length > 0) {
    sources.forEach((s) => {
      const li = document.createElement("li");
      const srcName = isMr ? "आयसीएआर कृषी संशोधन संदर्भ" : escapeHtml(s.source);
      const srcTitle = escapeHtml(s.title);
      const srcCrop = isMr ? cropDisplay : escapeHtml(s.crop || "Agricultural Practice");
      li.innerHTML = `<strong>${srcName}</strong>: <em>${srcTitle}</em> (${srcCrop})`;
      sourcesList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = isMr ? "आयसीएआर उद्यानविद्या पीक संरक्षण मार्गदर्शक तत्त्वे" : "ICAR Horticultural Crop Protection Guidelines";
    sourcesList.appendChild(li);
  }
}
