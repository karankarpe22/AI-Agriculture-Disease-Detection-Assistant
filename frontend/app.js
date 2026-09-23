/**
 * AI Agriculture Assistant (कृषी सहाय्यक)
 * Frontend Application Controller with Full Multilingual, Standalone & Follow-Up Q&A Capabilities
 */

const API_BASE = "";

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

// Application Initialization
document.addEventListener("DOMContentLoaded", () => {
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

// Mode Switching (Leaf Diagnosis vs Standalone Q&A)
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

  document.getElementById("txt-explanation-title").textContent = t.explanationTitle;
  document.getElementById("txt-management-title").textContent = t.managementTitle;
  document.getElementById("txt-prevention-title").textContent = t.preventionTitle;
  document.getElementById("txt-precautions-title").textContent = t.precautionsTitle;
  document.getElementById("txt-advisory-title").textContent = t.advisoryTitle;
  document.getElementById("txt-gradcam-title").textContent = t.gradcamTitle;
  document.getElementById("txt-gradcam-desc").textContent = t.gradcamDesc;
  document.getElementById("txt-sources-title").textContent = t.sourcesTitle;
  document.getElementById("txt-footer").textContent = t.footerText;

  document.getElementById("txt-followup-title").textContent = t.followupTitle;
  document.getElementById("txt-followup-sub").textContent = t.followupSub;
  followupInput.placeholder = t.followupPlaceholder;
  txtBtnFollowup.textContent = t.btnFollowup;

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

// Sample Selectors
function setupSampleSelectors() {
  document.querySelectorAll(".sample-pill").forEach((pill) => {
    pill.addEventListener("click", async () => {
      const src = pill.dataset.src;
      try {
        const res = await fetch(src);
        if (!res.ok) throw new Error("Sample image not found on server.");
        const blob = await res.blob();
        const file = new File([blob], pill.dataset.name + ".jpg", { type: "image/jpeg" });
        handleSelectedFile(file);
      } catch (err) {
        console.warn("Could not load sample file:", err);
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
      standaloneQuestionInput.value = chip.dataset.q;
    });
  });

  document.querySelectorAll(".follow-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      followupInput.value = chip.dataset.q;
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
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
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
      alert(`Analysis Error: ${error.message}`);
    } finally {
      analyzeBtn.disabled = false;
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
      alert(`Q&A Advisory Error: ${error.message}`);
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
        <div class="thread-ai-body">${escapeHtml(directAns)}</div>
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

// Render Results to UI (Shared by Initial Analysis & Standalone Q&A)
function renderResults(data, options = {}) {
  const { isStandalone = false, initialQuestion = "" } = options;

  emptyState.style.display = "none";
  resultsContent.style.display = "block";

  // Check if image quality failed (only applicable in leaf diagnosis)
  if (!isStandalone && !data.success) {
    qualityAlert.style.display = "flex";
    qualityAlertText.textContent = data.warning || "Image quality issue detected.";
    diagnosisBanner.style.display = "none";
    weatherBox.style.display = "none";
    guidanceSections.style.display = "none";
    qaAnswerBox.style.display = "none";
    gradcamBox.style.display = "none";
    followupCard.style.display = "none";
    return;
  }

  qualityAlert.style.display = "none";
  diagnosisBanner.style.display = "block";
  weatherBox.style.display = "block";
  guidanceSections.style.display = "flex";
  followupCard.style.display = "block";

  // 1. Direct Q&A Answer Card (Render if farmer asked a question or if direct_answer exists)
  const directAnswerText = data.direct_answer || data.guidance?.direct_answer || "";
  const displayQuestion = initialQuestion || data.question || data.farmer_question || "";

  if (directAnswerText && (displayQuestion || isStandalone)) {
    qaAnswerBox.style.display = "block";
    qaQuestionDisplay.textContent = displayQuestion ? `"${displayQuestion}"` : "Agricultural Inquiry";
    qaDirectAnswer.textContent = directAnswerText;
    currentDirectAnswerText = directAnswerText;
  } else if (directAnswerText) {
    qaAnswerBox.style.display = "block";
    qaQuestionDisplay.textContent = currentLanguage === "marathi" ? "तातडीची कृषी शिफारस" : "Immediate Agronomic Recommendation";
    qaDirectAnswer.textContent = directAnswerText;
    currentDirectAnswerText = directAnswerText;
  } else {
    qaAnswerBox.style.display = "none";
  }

  // 2. Diagnosis & Confidence Banner
  if (isStandalone) {
    resCrop.textContent = data.crop || "Agriculture Advisory";
    resDisease.textContent = data.disease || "Expert Q&A Guidance";
    resConfidence.textContent = "Verified";
    resConfidenceBox.style.display = "none";
    confBarTrack.style.display = "none";
    uncertaintyBox.style.display = "none";
  } else {
    resConfidenceBox.style.display = "flex";
    confBarTrack.style.display = "block";

    resCrop.textContent = data.crop || "Crop";
    resDisease.textContent = data.disease || "Healthy";
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
      uncertaintyText.textContent =
        data.warning || "Model confidence is below 60%. Please verify with a local agricultural officer.";
    } else {
      uncertaintyBox.style.display = "none";
    }

  }

  // 3. Weather Context
  const w = data.weather || {};
  weatherLoc.textContent = w.location || "Field";
  weatherDetails.textContent = `Temp: ${w.temperature_c ?? 25}°C | Humidity: ${w.humidity_percentage ?? 60}% | ${w.condition ?? "Seasonal"}`;
  weatherRiskText.textContent = w.risk_analysis || "Standard seasonal precautions apply.";

  // 4. Structured Guidance
  const g = data.guidance || {};
  guidanceExplanation.textContent = g.explanation || "Agricultural advisory analysis complete.";
  guidanceManagement.textContent = g.management_guidance || "Refer to evidence documents.";
  guidancePrevention.textContent = g.prevention || "Adopt recommended crop rotation.";
  guidancePrecautions.textContent = g.precautions || "Use standard safety equipment.";
  guidanceAdvisory.textContent = g.expert_advisory || "Consult your local KVK.";

  // Prepare full speech text
  currentGuidanceText = `${data.crop || "Crop"} ${data.disease || ""}. ${directAnswerText ? directAnswerText + ". " : ""}${g.explanation || ""}. ${g.management_guidance || ""}`;

  // 5. Grad-CAM (Only in image diagnosis mode)
  if (!isStandalone && data.gradcam_base64) {
    gradcamBox.style.display = "block";
    gradcamImage.src = `data:image/jpeg;base64,${data.gradcam_base64}`;
  } else {
    gradcamBox.style.display = "none";
  }

  // 6. Evidence Sources
  sourcesList.innerHTML = "";
  const sources = data.sources || [];
  if (sources.length > 0) {
    sources.forEach((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${escapeHtml(s.source)}</strong>: <em>${escapeHtml(s.title)}</em> (${escapeHtml(s.crop || "Agricultural Practice")})`;
      sourcesList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "ICAR Horticultural Crop Protection Guidelines";
    sourcesList.appendChild(li);
  }

  // Update follow-up subtitle with current context
  const cropDis = `${data.crop || "crop"}${data.disease ? " (" + data.disease + ")" : ""}`;
  document.getElementById("txt-followup-sub").textContent =
    currentLanguage === "marathi"
      ? `${cropDis} वरील फवारणी, मात्रा किंवा सेंद्रिय उपायांबाबत थेट शंका विचारा.`
      : `Ask specific doubts about ${cropDis}, treatment timing, or chemical alternatives.`;

  // Scroll smoothly to results
  resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
}
