const API_BASE = (window.location.protocol === 'file:' || window.location.port !== '8000') ? 'http://127.0.0.1:8000' : '';
/**
 * AI Agriculture Assistant - Frontend Application Logic
 * Multilingual UI (English / मराठी), Voice STT/TTS, and Pipeline Integration.
 */

// Translations Dictionary
const I18N = {
  english: {
    appTitle: "AI Agriculture Assistant",
    appSubtitle: "Evidence-Grounded Crop Disease Detection & Advisory",
    leafInput: "Crop Leaf Diagnosis",
    dropzonePrompt: 'Drag and drop a leaf image, or <span class="browse-link">browse</span>',
    dropzoneHint: "Supports clear photos of Tomato, Potato, Bell Pepper (JPG/PNG, max 10MB)",
    lblLocation: "📍 Farmer Location / City",
    lblQuestion: "❓ Farmer Question",
    lblQuestionSub: "(Optional specific question)",
    qPlaceholder: "e.g. What organic or chemical spray should I use? How do I stop this from spreading?",
    gradcamOpt: "Generate Grad-CAM Visual Attention Heatmap",
    btnAnalyze: "Analyze Crop Health",
    btnAnalyzing: "Analyzing Leaf & Weather...",
    diagnosisHeading: "Diagnosis & Agricultural Guidance",
    emptyTitle: "No Leaf Analyzed Yet",
    emptyDesc: "Upload a crop leaf image and click 'Analyze Crop Health' to diagnose disease, check local weather risk, and retrieve ICAR-grounded treatment advice.",
    confLabel: "Confidence",
    altProbabilities: "Alternative Model Probabilities",
    listenAdvisory: "Listen to Advisory",
    playingAudio: "Playing Audio...",
    explanationTitle: "Explanation",
    managementTitle: "Treatment & Management",
    preventionTitle: "Long-Term Prevention",
    precautionsTitle: "Precautions & Safety",
    advisoryTitle: "Expert Advisory",
    gradcamTitle: "Visual Explanation (Grad-CAM)",
    gradcamDesc: "Highlighted regions indicate leaf symptoms that influenced the MobileNetV3 model prediction:",
    sourcesTitle: "Agricultural Evidence Sources",
    footerText: "AI Agriculture Assistant • MobileNetV3-Large Classifier • RAG Knowledge Base • Gemini Guidance",
    sprayQ: "What spray should I use to treat this disease?",
    preventQ: "How can I prevent this in my other plants?",
    harvestQ: "Is this disease dangerous for fruit harvest?",
  },
  marathi: {
    appTitle: "कृषी सहाय्यक (AI Agriculture Assistant)",
    appSubtitle: "प्रमाणित कृषी पुरावे आणि हवामानावर आधारित पीक रोग निदान व सल्ला",
    leafInput: "पिकांच्या पानांचे रोग निदान",
    dropzonePrompt: 'पानाचा फोटो येथे ड्रॅग करा किंवा <span class="browse-link">निवडा</span>',
    dropzoneHint: "टोमॅटो, बटाटा, ढोबळी मिरचीच्या पानांचे स्पष्ट फोटो (JPG/PNG, कमाल १० MB)",
    lblLocation: "📍 शेतकऱ्याचे ठिकाण / शहर",
    lblQuestion: "❓ शेतकऱ्याचा प्रश्न",
    lblQuestionSub: "(पर्यायी विशिष्ट शंका)",
    qPlaceholder: "उदा. कोणती सेंद्रिय किंवा रासायनिक फवारणी करावी? रोगाचा प्रसार कसा रोखावा?",
    gradcamOpt: "Grad-CAM व्हिज्युअल हीटमॅप तयार करा",
    btnAnalyze: "पिकाचे आरोग्य तपासा",
    btnAnalyzing: "निदान व सल्ला सुरू आहे...",
    diagnosisHeading: "रोग निदान आणि कृषी सल्ला",
    emptyTitle: "अद्याप पानाचे विश्लेषण केलेले नाही",
    emptyDesc: "पानाचा फोटो अपलोड करा आणि 'पिकाचे आरोग्य तपासा' वर क्लिक करा. मॉडेल रोग शोधून हवामान व ICAR पुराव्यांवर आधारित मार्गदर्शन देईल.",
    confLabel: "विश्वासार्हता",
    altProbabilities: "इतर संभाव्य रोग शक्यता",
    listenAdvisory: "सल्ला ऐका (ऑडिओ)",
    playingAudio: "ऑडिओ वाजत आहे...",
    explanationTitle: "रोगाचे स्पष्टीकरण व कारणे",
    managementTitle: "उपचार व नियंत्रण पद्धती",
    preventionTitle: "दीर्घकालीन प्रतिबंधात्मक उपाय",
    precautionsTitle: "सुरक्षा व सावधगिरी",
    advisoryTitle: "तज्ज्ञ कृषी सल्ला (KVK)",
    gradcamTitle: "व्हिज्युअल स्पष्टीकरण (Grad-CAM)",
    gradcamDesc: "हायलाइट केलेले भाग MobileNetV3 मॉडेलने निदान करताना विचारात घेतलेली लक्षणे दर्शवतात:",
    sourcesTitle: "वापरलेले कृषी संशोधन पुरावे व संदर्भ",
    footerText: "कृषी सहाय्यक • MobileNetV3-Large क्लासिफायर • RAG नॉलेज बेस • जेमिनी मार्गदर्शन",
    sprayQ: "या रोगावर नियंत्रणासाठी कोणती फवारणी करावी?",
    preventQ: "इतर झाडांवर हा रोग पसरू नये म्हणून काय करावे?",
    harvestQ: "हा रोग फळांच्या उत्पादनासाठी किती घातक आहे?",
  },
};

let currentLanguage = "english";
let selectedFile = null;
let currentGuidanceText = "";

// DOM Elements
const langEnBtn = document.getElementById("lang-en");
const langMrBtn = document.getElementById("lang-mr");
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
const emptyState = document.getElementById("empty-state");
const resultsContent = document.getElementById("results-content");
const qualityAlert = document.getElementById("quality-alert");
const qualityAlertText = document.getElementById("quality-alert-text");
const resCrop = document.getElementById("res-crop");
const resDisease = document.getElementById("res-disease");
const resConfidence = document.getElementById("res-confidence");
const confBarFill = document.getElementById("conf-bar-fill");
const uncertaintyBox = document.getElementById("uncertainty-box");
const uncertaintyText = document.getElementById("uncertainty-text");
const topPredsList = document.getElementById("top-preds-list");
const weatherLoc = document.getElementById("weather-loc");
const weatherDetails = document.getElementById("weather-details");
const weatherRiskText = document.getElementById("weather-risk-text");
const ttsBtn = document.getElementById("tts-btn");
const guidanceAudio = document.getElementById("guidance-audio");
const guidanceExplanation = document.getElementById("guidance-explanation");
const guidanceManagement = document.getElementById("guidance-management");
const guidancePrevention = document.getElementById("guidance-prevention");
const guidancePrecautions = document.getElementById("guidance-precautions");
const guidanceAdvisory = document.getElementById("guidance-advisory");
const gradcamBox = document.getElementById("gradcam-box");
const gradcamImage = document.getElementById("gradcam-image");
const sourcesList = document.getElementById("sources-list");
const systemBadge = document.getElementById("system-status-text");

// Initialize application
document.addEventListener("DOMContentLoaded", () => {
  setupLanguageSwitcher();
  setupDropzone();
  setupQuickChips();
  setupSpeechRecognition();
  setupTTS();
  setupFormSubmission();
  checkSystemHealth();
});

// System Health Check
async function checkSystemHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      systemBadge.textContent = `${data.model} Ready (${data.num_classes} classes)`;
    }
  } catch (e) {
    systemBadge.textContent = "Offline Mode";
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
  document.getElementById("txt-leaf-input").textContent = t.leafInput;
  document.getElementById("txt-dropzone-prompt").innerHTML = t.dropzonePrompt;
  document.getElementById("txt-dropzone-hint").textContent = t.dropzoneHint;
  document.getElementById("lbl-location").children[0].textContent = t.lblLocation;
  document.getElementById("lbl-question").children[0].textContent = t.lblQuestion;
  document.getElementById("lbl-question").children[1].textContent = t.lblQuestionSub;
  questionInput.placeholder = t.qPlaceholder;
  document.getElementById("txt-gradcam-opt").textContent = t.gradcamOpt;
  btnAnalyzeText.textContent = t.btnAnalyze;
  document.getElementById("txt-diagnosis-heading").textContent = t.diagnosisHeading;
  document.getElementById("txt-empty-title").textContent = t.emptyTitle;
  document.getElementById("txt-empty-desc").textContent = t.emptyDesc;
  document.getElementById("txt-conf-label").textContent = t.confLabel;
  document.getElementById("txt-other-possibilities").textContent = t.altProbabilities;
  document.getElementById("txt-listen-guidance").textContent = t.listenAdvisory;
  document.getElementById("txt-explanation-title").textContent = t.explanationTitle;
  document.getElementById("txt-management-title").textContent = t.managementTitle;
  document.getElementById("txt-prevention-title").textContent = t.preventionTitle;
  document.getElementById("txt-precautions-title").textContent = t.precautionsTitle;
  document.getElementById("txt-advisory-title").textContent = t.advisoryTitle;
  document.getElementById("txt-gradcam-title").textContent = t.gradcamTitle;
  document.getElementById("txt-gradcam-desc").textContent = t.gradcamDesc;
  document.getElementById("txt-sources-title").textContent = t.sourcesTitle;
  document.getElementById("txt-footer").textContent = t.footerText;

  // Update question quick chips
  const qChips = document.querySelectorAll(".q-chip");
  if (qChips.length >= 3) {
    qChips[0].textContent = lang === "marathi" ? "फवारणी सल्ला?" : "Spray guidance?";
    qChips[0].dataset.q = t.sprayQ;
    qChips[1].textContent = lang === "marathi" ? "प्रतिबंध?" : "Prevention?";
    qChips[1].dataset.q = t.preventQ;
    qChips[2].textContent = lang === "marathi" ? "उत्पादन धोका?" : "Harvest risk?";
    qChips[2].dataset.q = t.harvestQ;
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
  };
  reader.readAsDataURL(file);
}

function resetFileInput() {
  selectedFile = null;
  leafInput.value = "";
  imagePreview.src = "";
  previewContainer.style.display = "none";
  dropzonePrompt.style.display = "block";
  analyzeBtn.disabled = true;
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
}

// Speech Recognition (Web Speech API)
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.style.display = "none";
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = false;

  micBtn.addEventListener("click", () => {
    recognition.lang = currentLanguage === "marathi" ? "mr-IN" : "en-IN";
    try {
      recognition.start();
      micBtn.classList.add("recording");
    } catch (e) {
      recognition.stop();
      micBtn.classList.remove("recording");
    }
  });

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    questionInput.value = transcript;
    micBtn.classList.remove("recording");
  };

  recognition.onerror = () => {
    micBtn.classList.remove("recording");
  };

  recognition.onend = () => {
    micBtn.classList.remove("recording");
  };
}

// Text-to-Speech playback
function setupTTS() {
  ttsBtn.addEventListener("click", async () => {
    if (!currentGuidanceText) return;

    ttsBtn.disabled = true;
    const origText = ttsBtn.querySelector("span:last-child").textContent;
    ttsBtn.querySelector("span:last-child").textContent = I18N[currentLanguage].playingAudio;

    try {
      const formData = new FormData();
      formData.append("text", currentGuidanceText);
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
        // Browser fallback SpeechSynthesis
        if ("speechSynthesis" in window) {
          const utter = new SpeechSynthesisUtterance(currentGuidanceText);
          utter.lang = currentLanguage === "marathi" ? "mr-IN" : "en-US";
          window.speechSynthesis.speak(utter);
        }
      }
    } catch (e) {
      console.warn("TTS fetch failed, attempting browser synth", e);
      if ("speechSynthesis" in window) {
        const utter = new SpeechSynthesisUtterance(currentGuidanceText);
        utter.lang = currentLanguage === "marathi" ? "mr-IN" : "en-US";
        window.speechSynthesis.speak(utter);
      }
    } finally {
      ttsBtn.disabled = false;
      ttsBtn.querySelector("span:last-child").textContent = origText;
    }
  });
}

// Form Submission & Pipeline Execution
function setupFormSubmission() {
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Set loading state
    analyzeBtn.disabled = true;
    analyzeSpinner.style.display = "inline-block";
    btnAnalyzeText.textContent = I18N[currentLanguage].btnAnalyzing;
    qualityAlert.style.display = "none";

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("question", questionInput.value.trim());
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

      renderResults(data);
    } catch (error) {
      alert(`Analysis Error: ${error.message}`);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.style.display = "none";
      btnAnalyzeText.textContent = I18N[currentLanguage].btnAnalyze;
    }
  });
}

// Render Results to UI
function renderResults(data) {
  emptyState.style.display = "none";
  resultsContent.style.display = "block";

  // Check if image quality failed
  if (!data.success) {
    qualityAlert.style.display = "flex";
    qualityAlertText.textContent = data.warning || "Image quality issue detected.";
    document.getElementById("diagnosis-banner").style.display = "none";
    document.getElementById("weather-box").style.display = "none";
    document.getElementById("guidance-sections").style.display = "none";
    return;
  }

  // Restore sections if previously hidden
  document.getElementById("diagnosis-banner").style.display = "block";
  document.getElementById("weather-box").style.display = "block";
  document.getElementById("guidance-sections").style.display = "flex";

  // 1. Diagnosis & Confidence
  resCrop.textContent = data.crop || "Crop";
  resDisease.textContent = data.disease || "Healthy";
  const confPct = (data.confidence * 100).toFixed(1);
  resConfidence.textContent = `${confPct}%`;
  confBarFill.style.width = `${Math.min(100, Math.max(10, data.confidence * 100))}%`;

  // Color-code confidence bar
  if (data.confidence >= 0.8) {
    confBarFill.style.background = "var(--primary)";
  } else if (data.confidence >= 0.6) {
    confBarFill.style.background = "var(--accent-amber)";
  } else {
    confBarFill.style.background = "var(--accent-red)";
  }

  // Uncertainty Warning
  if (data.is_low_confidence || data.confidence < 0.6) {
    uncertaintyBox.style.display = "flex";
    uncertaintyText.textContent =
      data.warning || "Model confidence is below 60%. Please verify with a local agricultural officer.";
  } else {
    uncertaintyBox.style.display = "none";
  }

  // Top Predictions
  topPredsList.innerHTML = "";
  if (data.top_predictions && data.top_predictions.length > 0) {
    data.top_predictions.forEach((p) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${p.crop} - ${p.disease}</span> <strong>${(p.confidence * 100).toFixed(1)}%</strong>`;
      topPredsList.appendChild(li);
    });
  }

  // 2. Weather Context
  const w = data.weather || {};
  weatherLoc.textContent = w.location || locationInput.value || "Local Farm";
  weatherDetails.textContent = `Temp: ${w.temperature_c ?? 25}°C | Humidity: ${w.humidity_percentage ?? 60}% | ${w.condition ?? "Seasonal"}`;
  weatherRiskText.textContent = w.risk_analysis || "Standard seasonal precautions apply.";

  // 3. Structured Guidance
  const g = data.guidance || {};
  guidanceExplanation.textContent = g.explanation || "Diagnostic analysis complete.";
  guidanceManagement.textContent = g.management_guidance || "Refer to evidence documents.";
  guidancePrevention.textContent = g.prevention || "Adopt recommended crop rotation.";
  guidancePrecautions.textContent = g.precautions || "Use standard safety equipment.";
  guidanceAdvisory.textContent = g.expert_advisory || "Consult your local KVK.";

  // Combine guidance text for TTS
  currentGuidanceText = `${data.crop} ${data.disease}. ${g.explanation}. ${g.management_guidance}`;

  // 4. Grad-CAM
  if (data.gradcam_base64) {
    gradcamBox.style.display = "block";
    gradcamImage.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
  } else {
    gradcamBox.style.display = "none";
  }

  // 5. Evidence Sources
  sourcesList.innerHTML = "";
  const sources = data.sources || [];
  if (sources.length > 0) {
    sources.forEach((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${s.source}</strong>: <em>${s.title}</em> (${s.crop || "Agricultural Practice"})`;
      sourcesList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "ICAR Horticultural Crop Protection Guidelines";
    sourcesList.appendChild(li);
  }

  // Scroll smoothly to results
  resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
}
