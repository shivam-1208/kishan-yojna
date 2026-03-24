# =============================================================================
# Kisan Saathi — Backend Server  (LLM + Keyword Hybrid Chatbot)
#
# CHAT STRATEGY (3-tier priority):
#   Tier 1 — Diagnosis context  : if user asks about a just-detected disease,
#             return precise treatment from DEFICIENCY_DATA immediately.
#             Never waste an LLM call on something we know exactly.
#   Tier 2 — Keyword match      : 20 agronomy topics. Fast, offline, free.
#             Catches ~70% of farmer questions. No API cost.
#   Tier 3 — Claude LLM         : anything Tier 1/2 can't answer confidently.
#             Sends a rich system prompt containing all DEFICIENCY_DATA and
#             CHAT_KNOWLEDGE so the LLM acts as a grounded agronomist,
#             not a hallucinating general assistant.
#
# ENDPOINTS:
#   POST /api/predict        — MSTC-Net inference (TTA, LBP maps)
#   POST /api/chat           — Hybrid chatbot
#   POST /api/chat/stream    — Streaming LLM response (SSE)
#   GET  /api/health         — Status
#   GET/POST /api/history    — Scan history
#   DELETE /api/history/<id> — Delete record
#   GET  /api/connect        — Government contacts
#   GET  /api/tips           — Seasonal tips
#   GET  /api/deficiency/<k> — Deficiency detail
#   GET  /api/weather        — Weather (OpenWeatherMap)
#   GET  /                   — Serve frontend
#
# ENV VARS:
#   GROQ_API_KEY             — Required for Tier 3 LLM (free at console.groq.com)
#   OPENWEATHER_API_KEY      — Optional, for real weather
# =============================================================================

import os, io, cv2, json, time, copy, logging, traceback, warnings
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Suppress noisy but non-fatal warnings ────────────────────────────────────
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Load .env if available ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths & config ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

MODEL_PATH    = Path(os.getenv("MODEL_PATH",    str(BASE_DIR / "best_mtga_efficientnet.pth")))
FRONTEND_PATH = Path(os.getenv("FRONTEND_PATH", str(BASE_DIR / "kisan_saathi.html")))
HISTORY_FILE  = Path(os.getenv("HISTORY_FILE",  str(BASE_DIR / "scan_history.json")))

# ML pipeline file paths (relative to project root or set via env)
MODELS_DIR        = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
SCALER_PATH       = MODELS_DIR / "scaler.pkl"
XGB_PATH          = MODELS_DIR / "XGBoost.pkl"
LABEL_ENCODER_PATH= MODELS_DIR / "label_encoder (1).pkl"

CLASS_NAMES = ["N", "K", "Mg", "healthy"]
IMG_SIZE    = 224
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
LBP_RADII   = [1, 2, 3]
LBP_POINTS  = [8, 16, 24]
BACKBONE    = os.getenv("BACKBONE", "resnet").lower()

GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

log.info(f"Device: {DEVICE} | Groq SDK: {GROQ_OK} | Key: {'set' if GROQ_KEY else 'missing'}")

# =============================================================================
# TENSORFLOW / MOBILENETV2  (feature extractor for XGBoost pipeline)
# =============================================================================
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    _cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    TF_OK = True
    log.info("✅ MobileNetV2 loaded (1280-dim features)")
except Exception as e:
    _cnn = None
    TF_OK = False
    log.warning(f"TensorFlow not available: {e}")

# =============================================================================
# ML PIPELINE MANAGER  (XGBoost + StandardScaler + LabelEncoder)
# =============================================================================
class MLManager:
    """
    Loads scaler.pkl, XGBoost.pkl, label_encoder (1).pkl separately.
    Paths are constructed first; joblib.load() is only called after
    confirming each file exists — avoids the AttributeError seen when
    joblib.load() was accidentally called during path assignment.
    """
    def __init__(self):
        self.scaler = None
        self.model  = None   # XGBoost classifier
        self.le     = None   # LabelEncoder
        self.loaded = False
        self._load()

    def _load(self):
        if not JOBLIB_OK:
            log.warning("joblib not installed — pip install joblib")
            return

        # ✅ FIX: define Path objects first; never call joblib.load() here
        paths = {
            "scaler":        SCALER_PATH,
            "xgb":           XGB_PATH,
            "label_encoder": LABEL_ENCODER_PATH,
        }

        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            log.warning(f"ML files missing: {missing}")
            log.warning(f"Copy scaler.pkl, XGBoost.pkl, 'label_encoder (1).pkl' "
                        f"to: {MODELS_DIR}")
            return

        try:
            self.scaler = joblib.load(str(paths["scaler"]))
            self.model  = joblib.load(str(paths["xgb"]))
            self.le     = joblib.load(str(paths["label_encoder"]))
            self.loaded = True
            log.info(f"✅ ML pipeline loaded from {MODELS_DIR}")
        except Exception as e:
            log.error(f"ML load failed: {e}")
            traceback.print_exc()

    @property
    def ready(self):
        return self.loaded and self.model is not None and self.scaler is not None


ml_manager = MLManager()

# =============================================================================
# FEATURE EXTRACTION  (matches Kaggle training code exactly)
# =============================================================================
def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    MobileNetV2 GAP features — 1280-dim vector.
    Input: BGR numpy array (uint8).
    """
    img = cv2.resize(img_bgr, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    feat = _cnn.predict(img, verbose=0)[0]   # shape (1280,)
    return feat


def predict_ml(pil_img: Image.Image) -> dict:
    """
    Full ML inference pipeline:
      PIL → BGR → extract_features (1280-dim)
      → scaler.transform → xgb.predict + predict_proba
    """
    if not ml_manager.ready:
        raise RuntimeError("ML models not loaded — check scaler.pkl / XGBoost.pkl / label_encoder.pkl")
    if not TF_OK:
        raise RuntimeError("TensorFlow not installed — pip install tensorflow")

    # PIL → BGR (matches cv2.imread used in training)
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    feat        = extract_features(img_bgr)
    feat_scaled = ml_manager.scaler.transform([feat])
    pred        = ml_manager.model.predict(feat_scaled)[0]
    proba       = ml_manager.model.predict_proba(feat_scaled)[0]
    label       = ml_manager.le.inverse_transform([pred])[0]

    # Build probs in CLASS_NAMES order
    probs = []
    for cn in CLASS_NAMES:
        try:
            idx = list(ml_manager.le.classes_).index(cn)
            probs.append(float(proba[idx]))
        except ValueError:
            probs.append(0.0)

    return {
        "cls":         label,
        "label":       DEFICIENCY_DATA[label]["label"],
        "confidence":  round(float(np.max(proba)) * 100, 2),
        "probs":       [round(p, 4) for p in probs],
        "all_classes": CLASS_NAMES,
        "model_type":  "ml",
    }

# =============================================================================
# KNOWLEDGE BASE
# =============================================================================
DEFICIENCY_DATA = {
    "N": {
        "label": "Nitrogen Deficiency",
        "cause": "Insufficient nitrogen in soil, excessive rainfall leaching, poor root uptake.",
        "symptoms": "Uniform yellowing starting from older leaves, stunted growth, pale green colour.",
        "crops": "Wheat, Rice, Maize, Sugarcane",
        "severity": 65,
        "treatments": [
            "Apply Urea @ 50 kg/acre as top dressing",
            "Use DAP (Di-Ammonium Phosphate) at sowing",
            "Foliar spray: 2% Urea solution every 10 days",
            "Add organic compost or green manure",
            "Test soil pH — keep between 6.0–7.5 for best N uptake",
        ],
        "prevention": "Split nitrogen into 2–3 doses. Avoid waterlogging.",
        "organic_alt": "Vermicompost 2 t/acre, green manure (Dhaincha), farmyard manure 5 t/acre.",
    },
    "K": {
        "label": "Potassium Deficiency",
        "cause": "Sandy soils with low K retention, high rainfall, imbalanced fertilization.",
        "symptoms": "Brown scorching and curling of leaf edges starting from older leaves.",
        "crops": "Potato, Tomato, Banana, Cotton",
        "severity": 55,
        "treatments": [
            "Apply MOP (Muriate of Potash) @ 40 kg/acre",
            "Foliar spray: 0.5% KNO₃ solution",
            "Use SOP (Sulphate of Potash) on sensitive crops",
            "Incorporate potassium-rich organic matter",
            "Avoid excess nitrogen which competes with K uptake",
        ],
        "prevention": "Regular soil testing every 2 years. Apply K before sowing.",
        "organic_alt": "Banana peel compost, wood ash (contains 5–7% K), granite dust.",
    },
    "Mg": {
        "label": "Magnesium Deficiency",
        "cause": "Acidic soils, excessive K or Ca blocking Mg, leaching in sandy soils.",
        "symptoms": "Interveinal chlorosis — yellowing between green veins, middle leaves first.",
        "crops": "Rice, Groundnut, Citrus, Coffee",
        "severity": 45,
        "treatments": [
            "Apply Magnesium Sulphate (MgSO₄) @ 25 kg/acre",
            "Foliar spray: 2% MgSO₄ solution 2–3 times",
            "Apply dolomite lime if soil is acidic",
            "Reduce excessive K fertilization",
            "Use chelated Mg micronutrient mixture",
        ],
        "prevention": "Maintain soil pH above 6.0. Avoid heavy K without Mg.",
        "organic_alt": "Dolomite lime, compost from seaweed, Epsom salt spray (1 tbsp/litre).",
    },
    "healthy": {
        "label": "Healthy Plant",
        "cause": "No stress detected. Plant is well-nourished.",
        "symptoms": "Uniform green, no chlorosis, good leaf structure.",
        "crops": "All crops",
        "severity": 5,
        "treatments": [
            "Continue current fertilization schedule",
            "Monitor soil moisture regularly",
            "Apply preventive fungicide if weather is humid",
            "Maintain proper plant spacing for air circulation",
            "Schedule next soil test in 3 months",
        ],
        "prevention": "Regular monitoring, balanced nutrition, timely irrigation.",
        "organic_alt": "Maintain organic matter levels with compost and mulching.",
    },
}

# Keyword → answer mapping (Tier 2 — fast offline responses)
KEYWORD_KB = {
    "nitrogen":   "Nitrogen deficiency causes uniform yellowing from older leaves. Apply Urea 50 kg/acre or DAP at sowing. Split into 2–3 doses. Avoid waterlogging which causes N loss.",
    "potassium":  "Potassium deficiency causes brown edges on leaves. Apply MOP 40 kg/acre. Foliar spray 0.5% KNO₃ works quickly. Sandy soils need more frequent K applications.",
    "magnesium":  "Magnesium deficiency shows yellowing between green veins. Spray 2% MgSO₄ 2–3 times. Apply dolomite if soil is acidic. Reduce excess K which blocks Mg.",
    "phosphorus": "Phosphorus deficiency causes purple/red tints on leaves and poor root development. Apply SSP (Single Super Phosphate) 100 kg/acre at sowing. Maintain soil pH 6–7 for best P uptake.",
    "zinc":       "Zinc deficiency causes khaira disease in rice — white/brown patches. Apply Zinc Sulphate 25 kg/acre to soil, or foliar spray 0.5% ZnSO₄. Common in alkaline soils.",
    "fertilizer": "For cereals use NPK 120:60:40 kg/ha. Always do a soil test first. Split urea into 3 doses: basal (30%), tillering (40%), panicle (30%). Over-fertilization wastes money and pollutes groundwater.",
    "organic":    "Organic options: vermicompost 2 t/acre, FYM 5 t/acre, neem cake 200 kg/acre, green manure (Dhaincha). Organic matter improves soil structure, water retention, and microbial activity.",
    "wheat":      "Wheat: N 120 kg/ha, P 60 kg/ha, K 40 kg/ha. Sow by 25 November. First irrigation at 21 DAS. Watch for yellow rust — spray Propiconazole 0.1% if rust appears on leaves.",
    "paddy":      "Paddy: N 120 kg/ha in 3 splits, P 60 kg/ha, K 40 kg/ha. Transplant at 25–30 DAS. Maintain 2–5 cm water. Apply ZnSO₄ 25 kg/ha if yellowing. Use light traps for BPH.",
    "maize":      "Maize: N 120–150 kg/ha in 3 splits, P 60 kg/ha, K 40 kg/ha. Apply N at sowing, knee-high, and tasselling stages. Watch for fall armyworm — spray Emamectin benzoate 0.5 g/L.",
    "sugarcane":  "Sugarcane: N 250 kg/ha, P 60 kg/ha, K 100 kg/ha. Apply in 3 splits. Trash mulching conserves moisture and adds organic matter. Intercrop with potato or onion for extra income.",
    "tomato":     "Tomato: Use NPK 150:75:75 kg/ha. Stake plants at 30 cm height. Spray Mancozeb 2 g/L for early blight. Drip irrigation with fertigation gives best yield.",
    "pest":       "For pest management: use yellow sticky traps for whiteflies, pheromone traps for stem borer. Neem oil 3 ml/L for minor infestations. For severe cases call Kisan helpline 1800-180-1551.",
    "spray":      "Best spray time: early morning 6–9 AM or evening 4–6 PM. Avoid spraying in direct sunlight or before rain. Always wear gloves, mask, and goggles. Use calibrated sprayer.",
    "soil":       "Soil testing every 2–3 years. Send 500g soil from 0–15 cm depth to nearest lab. Cost ₹5–25. Results guide exact doses. Ideal crop soil: pH 6–7, organic matter >0.8%, EC <1 dS/m.",
    "irrigation": "Drip irrigation saves 40–60% water. Install tensiometers to monitor moisture. Irrigate wheat at: crown root initiation (21 DAS), tillering, jointing, flowering, and grain fill.",
    "kisan":      "Kisan Call Centre: 1800-180-1551 (free, 24×7). PM-KISAN: ₹6000/year at pmkisan.gov.in. Soil Health Card at KVK. PM Fasal Bima Yojana for crop insurance.",
    "scheme":     "Government schemes: PM-KISAN (₹6000/yr), PM Fasal Bima Yojana (crop insurance), Soil Health Card scheme, Pradhan Mantri Krishi Sinchai Yojana (irrigation subsidy), e-NAM for market access.",
    "fungus":     "For fungal diseases: spray Mancozeb 2 g/L or Carbendazim 1 g/L. Avoid overhead irrigation. Improve air circulation between plants. Remove and burn infected plant debris.",
    "weed":       "For weed management: spray Pendimethalin 3 L/ha pre-emergence, or 2,4-D 1 kg/ha post-emergence in wheat. Hand weeding at 20–25 DAS is most effective in small fields.",
    "namaste":    "Namaste! 🙏 I am Krishi Expert Bot. Ask me about fertilizers, crop diseases, pest management, or government schemes. I can also explain your plant stress scan results in detail.",
    "hello":      "Hello! Welcome to Kisan Saathi. I'm here to help with all your farming questions — from fertilizer doses to pest control. What would you like to know?",
}

# Topics where LLM adds clear value over keywords
LLM_TRIGGER_PHRASES = [
    "why", "explain", "what happens", "how does", "compare", "difference",
    "best time", "which fertilizer", "how much", "can i use", "is it safe",
    "climate change", "drought", "flood", "mixed crop", "intercrop",
    "market price", "export", "storage", "post harvest", "income",
    "disease spread", "resistant variety", "hybrid seed",
]

# =============================================================================
# LLM CLIENT  (Groq — free, fast inference)
# =============================================================================
def build_system_prompt(last_diagnosis: str = None) -> str:
    deficiency_summary = ""
    for cls, d in DEFICIENCY_DATA.items():
        treatments = "; ".join(d["treatments"])
        deficiency_summary += (
            f"\n{d['label']}:\n"
            f"  Cause: {d['cause']}\n"
            f"  Symptoms: {d['symptoms']}\n"
            f"  Treatments: {treatments}\n"
            f"  Prevention: {d['prevention']}\n"
            f"  Organic alternatives: {d['organic_alt']}\n"
        )

    diagnosis_ctx = ""
    if last_diagnosis and last_diagnosis in DEFICIENCY_DATA:
        d = DEFICIENCY_DATA[last_diagnosis]
        diagnosis_ctx = (
            f"\n\nCURRENT SCAN CONTEXT: The farmer's scan just detected "
            f"'{d['label']}'. When the farmer asks about treatment, cure, spray, "
            f"or fertilizer, always prioritize advice for THIS specific deficiency."
        )

    return f"""You are Krishi Expert Bot, an AI agronomist assistant embedded in the 
Kisan Saathi plant stress detection app for Indian farmers.

YOUR ROLE:
- Give practical, specific, actionable advice for Indian farming conditions
- Use simple language — many farmers have limited formal education
- Include specific quantities (kg/acre, ml/litre, DAS) whenever relevant
- Mention government schemes and free resources when relevant
- Always suggest consulting local KVK or calling 1800-180-1551 for complex issues
- You can respond in Hindi/English mix (Hinglish) if the farmer writes in Hindi

KNOWLEDGE BASE — USE THIS, DO NOT DEVIATE:
{deficiency_summary}

KEYWORD KNOWLEDGE:
- Kisan helpline: 1800-180-1551 (free, 24x7)
- PM-KISAN: Rs 6000/year direct benefit transfer
- KVK: Krishi Vigyan Kendra, government extension centres across India
- e-NAM: Electronic National Agriculture Market for price discovery{diagnosis_ctx}

RULES:
1. Base your answers on the knowledge above. Never fabricate chemical names or doses.
2. If unsure, say so clearly and recommend KVK or helpline.
3. Keep answers concise — under 200 words unless detail is explicitly needed.
4. Format with short paragraphs or numbered steps. No long unbroken text.
5. Always end treatment advice with a safety reminder or preventive tip."""


def call_llm(message: str, history: list, last_diagnosis: str = None) -> str:
    if not GROQ_KEY or not GROQ_OK:
        return None
    try:
        client   = Groq(api_key=GROQ_KEY)
        messages = [{"role": "system", "content": build_system_prompt(last_diagnosis)}]
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"Groq call failed: {e}")
        return None


def stream_llm(message: str, history: list, last_diagnosis: str = None):
    if not GROQ_KEY or not GROQ_OK:
        reply = keyword_reply(message, last_diagnosis) or generic_fallback()
        yield f"data: {json.dumps({'text': reply, 'done': False, 'source': 'keyword'})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True, 'source': 'keyword'})}\n\n"
        return
    try:
        client   = Groq(api_key=GROQ_KEY)
        messages = [{"role": "system", "content": build_system_prompt(last_diagnosis)}]
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.4,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'text': delta.content, 'done': False, 'source': 'llm'})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True, 'source': 'llm'})}\n\n"
    except Exception as e:
        log.error(f"Groq stream failed: {e}")
        fallback = keyword_reply(message, last_diagnosis) or generic_fallback()
        yield f"data: {json.dumps({'text': fallback, 'done': False, 'source': 'keyword'})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True, 'source': 'keyword'})}\n\n"

# =============================================================================
# HYBRID CHAT ENGINE
# =============================================================================
def keyword_reply(message: str, last_diagnosis: str = None) -> str:
    msg_lower = message.lower()

    # Tier 1 — diagnosis context
    if last_diagnosis and last_diagnosis in DEFICIENCY_DATA:
        treat_kws = ["treat", "cure", "fix", "remedy", "medicine",
                     "fertiliz", "spray", "apply", "how to help",
                     "kya karu", "kaise", "upay", "ilaj"]
        if any(kw in msg_lower for kw in treat_kws):
            d = DEFICIENCY_DATA[last_diagnosis]
            steps = "\n".join(f"{i+1}. {t}" for i, t in enumerate(d["treatments"]))
            return (
                f"Based on the detected **{d['label']}**, here are the treatment steps:\n\n"
                f"{steps}\n\n"
                f"🌱 **Organic alternative:** {d['organic_alt']}\n\n"
                f"🛡 **Prevention:** {d['prevention']}"
            )

    # Tier 2 — keyword matching
    for key, val in KEYWORD_KB.items():
        if key in msg_lower:
            return val

    return None  # escalate to LLM


def should_use_llm(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in LLM_TRIGGER_PHRASES) or len(message.split()) > 8


def generic_fallback() -> str:
    return (
        "That's a good question. For region-specific advice:\n\n"
        "• **Kisan Call Centre**: 1800-180-1551 (free, 24×7)\n"
        "• Visit your nearest Krishi Vigyan Kendra\n"
        "• Use the 'Get Help' tab in the app\n\n"
        "You can ask me about: fertilizers, pest control, wheat, paddy, "
        "organic farming, or government schemes."
    )

# =============================================================================
# DEEP LEARNING MODEL DEFINITION
# backbone.*  — ResNet50 with fc replaced by Linear(2048→512)
# head.0      — Flatten
# head.1      — BatchNorm1d(512)
# head.2      — SiLU
# head.3      — Dropout(0.4)
# head.4      — Linear(512→4)
# =============================================================================
def build_feature_extractor(name: str, mid_dim: int = 512):
    name = name.lower()
    if name in ("efficientnet", "efficientnet_b2"):
        b = models.efficientnet_b2(weights=None)
        b.classifier = nn.Linear(b.classifier[1].in_features, mid_dim)
        return b, mid_dim
    elif name == "efficientnet_b4":
        b = models.efficientnet_b4(weights=None)
        b.classifier = nn.Linear(b.classifier[1].in_features, mid_dim)
        return b, mid_dim
    elif name == "resnet":
        b = models.resnet50(weights=None)
        b.fc = nn.Linear(b.fc.in_features, mid_dim)
        return b, mid_dim
    elif name == "densenet":
        b = models.densenet121(weights=None)
        b.classifier = nn.Linear(b.classifier.in_features, mid_dim)
        return b, mid_dim
    elif name == "mobilenet":
        b = models.mobilenet_v3_large(weights=None)
        b.classifier = nn.Linear(b.classifier[0].in_features, mid_dim)
        return b, mid_dim
    else:
        raise ValueError(f"Unknown backbone: {name}")


class PlantNetV4(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 4, mid_dim: int = 512):
        super().__init__()
        backbone, out_dim = build_feature_extractor(backbone_name, mid_dim)
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, rgb, lbp_maps=None):
        x = self.backbone(rgb)
        if x.dim() > 2:
            x = x.mean(dim=[2, 3])
        logits = self.head(x)
        return {"logits": logits, "tex_vecs": []}


PlantModel = PlantNetV4

# =============================================================================
# DL MODEL LOADER
# =============================================================================
class ModelManager:
    def __init__(self):
        self.model  = None
        self.loaded = False
        self._load()

    def _load(self):
        mp = Path(MODEL_PATH)
        log.info(f"Loading: backbone={BACKBONE}, path={mp}")
        try:
            self.model = PlantModel(BACKBONE, num_classes=4).to(DEVICE)
            if not mp.exists():
                log.warning(f"Checkpoint not found: {mp}")
                log.warning("Running in DEMO MODE — predictions will be random")
                self.model.eval()
                self.loaded = True
                return

            raw   = torch.load(str(mp), map_location=DEVICE, weights_only=False)
            state = raw.state_dict() if hasattr(raw, 'state_dict') else raw

            if list(state.keys())[0].startswith("module."):
                state = {k[len("module."):]: v for k, v in state.items()}
                log.info("Stripped 'module.' DDP prefix")

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            loaded_n = len(state) - len(unexpected)
            total    = len(state)
            log.info(f"Weights: {loaded_n}/{total} loaded | "
                     f"{len(missing)} missing | {len(unexpected)} unexpected")

            if loaded_n < total * 0.9:
                log.error(f"Only {loaded_n}/{total} weights matched — architecture mismatch!")
            else:
                log.info(f"✅ Checkpoint loaded successfully ({loaded_n}/{total} weights)")

            self.model.eval()
            with torch.no_grad():
                out = self.model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE))
                log.info(f"✅ Warm-up OK — logits shape: {out['logits'].shape}")
            self.loaded = True

        except Exception as e:
            log.error(f"DL model load failed: {e}")
            traceback.print_exc()
            if self.model:
                self.model.eval()
                self.loaded = True

    @property
    def ready(self):
        return self.loaded and self.model is not None


model_manager = ModelManager()

# =============================================================================
# PREPROCESSING + DL INFERENCE
# =============================================================================
val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

TTA_TFS = [
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1), T.ToTensor(), T.Normalize(MEAN, STD)]),
    T.Compose([T.Resize((IMG_SIZE+24, IMG_SIZE+24)), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomVerticalFlip(p=1), T.ToTensor(), T.Normalize(MEAN, STD)]),
    T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomRotation((90, 90)), T.ToTensor(), T.Normalize(MEAN, STD)]),
]


def compute_lbp_maps(img_rgb: np.ndarray) -> list:
    if not SKIMAGE_OK:
        return [torch.zeros(1, 1, IMG_SIZE, IMG_SIZE).to(DEVICE)] * 3
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    maps = []
    for R, P in zip(LBP_RADII, LBP_POINTS):
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform").astype(np.float32)
        lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6)
        t   = torch.from_numpy(lbp[np.newaxis, np.newaxis])
        t   = F.interpolate(t, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        maps.append(t.to(DEVICE))
    return maps


def predict_dl(pil_img: Image.Image) -> dict:
    """Deep learning inference with TTA."""
    if not model_manager.ready:
        raise RuntimeError("DL model not loaded")
    img_np     = np.array(pil_img.convert("RGB"))
    lbp_maps   = compute_lbp_maps(img_np)
    logits_sum = None
    with torch.no_grad():
        for tf in TTA_TFS:
            rgb_t = tf(pil_img).unsqueeze(0).to(DEVICE)
            out   = model_manager.model(rgb_t, lbp_maps)
            logits_sum = out["logits"] if logits_sum is None else logits_sum + out["logits"]
    probs    = F.softmax(logits_sum, dim=1).squeeze().cpu().numpy().tolist()
    pred_idx = int(np.argmax(probs))
    cls      = CLASS_NAMES[pred_idx]
    return {
        "cls":         cls,
        "label":       DEFICIENCY_DATA[cls]["label"],
        "confidence":  round(probs[pred_idx] * 100, 2),
        "probs":       [round(p, 4) for p in probs],
        "all_classes": CLASS_NAMES,
        "model_type":  "dl",
    }


def predict_single(pil_img: Image.Image) -> dict:
    """Route prediction: ML pipeline if ready, else DL model."""
    if ml_manager.ready and TF_OK:
        return predict_ml(pil_img)
    if model_manager.ready:
        log.warning("ML not ready — falling back to DL model")
        return predict_dl(pil_img)
    raise RuntimeError(
        "No predictor ready. Ensure scaler.pkl / XGBoost.pkl / label_encoder.pkl exist "
        "in the models/ folder and tensorflow/joblib are installed."
    )

# =============================================================================
# SCAN HISTORY
# =============================================================================
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def append_history(record: dict):
    h = load_history()
    h.insert(0, record)
    HISTORY_FILE.write_text(json.dumps(h[:100], indent=2))

# =============================================================================
# GOVERNMENT CONTACTS + SEASONAL TIPS
# =============================================================================
CONNECT_LIST = [
    {"icon":"🏛","label":"Kisan Call Centre","org":"Ministry of Agriculture","desc":"24×7 free helpline for farmers.","tag":"Free","phone":"1800-180-1551"},
    {"icon":"🌾","label":"Krishi Vigyan Kendra","org":"ICAR Network","desc":"Local extension centre. Soil testing, free seeds.","tag":"Govt","phone":""},
    {"icon":"💊","label":"Agri Input Dealer","org":"Certified Retailer","desc":"Certified fertilizer and pesticide dealer.","tag":"Nearby","phone":""},
    {"icon":"📱","label":"Digital India Agri","org":"e-NAM Platform","desc":"Sell produce at best prices nationwide.","tag":"Free","phone":""},
    {"icon":"🚨","label":"Pest Alert System","org":"NCIPM","desc":"Real-time pest and disease alerts.","tag":"24h","phone":""},
    {"icon":"💰","label":"PM Kisan Yojana","org":"Govt of India","desc":"Check PM-KISAN installment status.","tag":"Govt","phone":""},
    {"icon":"🌊","label":"Water Management","org":"Minor Irrigation Dept","desc":"Drip irrigation subsidy and water audit.","tag":"Subsidy","phone":""},
    {"icon":"🧑‍🌾","label":"Local Farmer Group","org":"FPO Network","desc":"Join an FPO for collective bargaining.","tag":"Free","phone":""},
]

TIPS_LIST = [
    {"season":"🌾 Rabi Season (Oct–Mar)","crop":"Wheat","tips":["Sow by 25 Nov for best yield","Irrigate at crown root initiation (21 DAS)","Apply 120 kg N/ha in 3 splits","Watch for yellow rust — spray Propiconazole"]},
    {"season":"🌱 Kharif Season (Jun–Sep)","crop":"Paddy","tips":["Transplant 25–30 day old seedlings","Maintain 2–5 cm water level first 4 weeks","Apply zinc sulphate @ 25 kg/ha if yellowing","Use pheromone traps for stem borer"]},
    {"season":"🍅 Year-Round","crop":"Vegetables","tips":["Mulching reduces water use by 40%","Fertigation with drip saves 30% fertilizer","Yellow sticky traps for whitefly and thrips","Spray neem oil (3 ml/L) as eco-friendly pesticide"]},
]

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/")
def index():
    fp = Path(FRONTEND_PATH)
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    for alt in [BASE_DIR / "templates" / "index.html",
                BASE_DIR / "templates" / "kisan_saathi.html",
                BASE_DIR / "index.html"]:
        if alt.exists():
            return alt.read_text(encoding="utf-8")
    return (
        "<h2>Frontend not found.</h2>"
        f"<p>Looked for: <code>{fp}</code></p>"
        "<p>Put <code>kisan_saathi.html</code> in the same folder as <code>app.py</code>, "
        "or set <code>FRONTEND_PATH</code> in your <code>.env</code> file.</p>"
    ), 404


@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "ml_loaded":    ml_manager.ready,
        "tf_loaded":    TF_OK,
        "dl_loaded":    model_manager.ready,
        "active_model": (
            "XGBoost (MobileNetV2 features)" if ml_manager.ready
            else ("DL (ResNet/EfficientNet)" if model_manager.ready else "none")
        ),
        "device":       DEVICE,
        "llm_enabled":  bool(GROQ_KEY and GROQ_OK),
        "llm_model":    GROQ_MODEL if GROQ_KEY else "keyword-only",
        "timestamp":    datetime.now().isoformat(),
    })


# ── PREDICT ───────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        w, h = pil_img.size
        if max(w, h) > 1920:
            s = 1920 / max(w, h)
            pil_img = pil_img.resize((int(w*s), int(h*s)), Image.LANCZOS)

        t0     = time.time()
        result = predict_single(pil_img)
        result["deficiency_info"] = DEFICIENCY_DATA[result["cls"]]
        result["elapsed_ms"]      = round((time.time()-t0)*1000)
        result["timestamp"]       = datetime.now().isoformat()

        log.info(f"Predict [{result['model_type']}] -> "
                 f"{result['label']} {result['confidence']}% in {result['elapsed_ms']}ms")
        return jsonify(result)
    except Exception as e:
        log.error(f"Predict error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── CHAT ──────────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    last_dx = body.get("last_diagnosis")
    history = body.get("history", [])
    if not message:
        return jsonify({"error": "Empty message"}), 400

    kw_reply = keyword_reply(message, last_dx)
    if kw_reply and not should_use_llm(message):
        return jsonify({"reply": kw_reply, "source": "keyword"})

    llm_reply = call_llm(message, history, last_dx)
    if llm_reply:
        return jsonify({"reply": llm_reply, "source": "llm"})

    reply = kw_reply or generic_fallback()
    return jsonify({"reply": reply, "source": "keyword" if kw_reply else "fallback"})


# ── CHAT STREAM (SSE) ─────────────────────────────────────────────────────────
@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    last_dx = body.get("last_diagnosis")
    history = body.get("history", [])
    if not message:
        return jsonify({"error": "Empty message"}), 400

    kw_reply = keyword_reply(message, last_dx)
    if kw_reply and not should_use_llm(message):
        def kw_gen():
            yield f"data: {json.dumps({'text': kw_reply, 'done': False, 'source': 'keyword'})}\n\n"
            yield f"data: {json.dumps({'text': '', 'done': True, 'source': 'keyword'})}\n\n"
        return Response(stream_with_context(kw_gen()),
                        mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def llm_gen():
        for chunk in stream_llm(message, history, last_dx):
            yield chunk
    return Response(stream_with_context(llm_gen()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify(load_history())


@app.route("/api/history", methods=["POST"])
def post_history():
    body = request.get_json(force=True, silent=True) or {}
    if not {"cls", "label", "confidence"}.issubset(body):
        return jsonify({"error": "Missing fields"}), 400
    record = {
        "id":         int(time.time()*1000),
        "cls":        body["cls"],
        "label":      body["label"],
        "confidence": body["confidence"],
        "date":       datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "thumbnail":  body.get("thumbnail", ""),
    }
    append_history(record)
    return jsonify({"status": "saved", "record": record}), 201


@app.route("/api/history/<int:rid>", methods=["DELETE"])
def delete_history(rid):
    h = [r for r in load_history() if r.get("id") != rid]
    HISTORY_FILE.write_text(json.dumps(h, indent=2))
    return jsonify({"status": "deleted"})


# ── STATIC DATA ───────────────────────────────────────────────────────────────
@app.route("/api/connect")
def connect():
    return jsonify(CONNECT_LIST)


@app.route("/api/tips")
def tips():
    return jsonify(TIPS_LIST)


@app.route("/api/deficiency/<cls_key>")
def deficiency_info(cls_key):
    if cls_key not in DEFICIENCY_DATA:
        return jsonify({"error": "Unknown class"}), 404
    return jsonify(DEFICIENCY_DATA[cls_key])


# ── WEATHER ───────────────────────────────────────────────────────────────────
@app.route("/api/weather")
def weather():
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    lat = request.args.get("lat", "26.8467")
    lon = request.args.get("lon", "80.9462")
    if not api_key:
        return jsonify({"temp":28,"humidity":68,"wind_speed":12,"uv_index":7,
                        "rain_pct":20,"condition":"Partly Cloudy","icon":"🌤",
                        "location":"Lucknow, UP","mock":True})
    try:
        import urllib.request
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&units=metric&appid={api_key}")
        with urllib.request.urlopen(url, timeout=5) as r:
            d = json.loads(r.read())
        return jsonify({
            "temp":       round(d["main"]["temp"]),
            "humidity":   d["main"]["humidity"],
            "wind_speed": round(d["wind"]["speed"] * 3.6),
            "condition":  d["weather"][0]["description"].title(),
            "location":   d["name"],
            "icon":       "🌤",
            "mock":       False,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 503


# ── ERROR HANDLERS ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):  return jsonify({"error": "Not found"}), 404

@app.errorhandler(413)
def too_large(e):  return jsonify({"error": "File too large (max 10MB)"}), 413

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    ml_status = (
        "✅ XGBoost + MobileNetV2" if (ml_manager.ready and TF_OK)
        else f"⚠️  Not ready — copy scaler.pkl / XGBoost.pkl / 'label_encoder (1).pkl' to {MODELS_DIR}"
    )
    llm_status = (
        f"✅ Groq ({GROQ_MODEL})" if (GROQ_KEY and GROQ_OK)
        else "⚠️  Keyword-only (set GROQ_API_KEY in .env)"
    )

    log.info("=" * 58)
    log.info("  Kisan Saathi — Backend")
    log.info(f"  Predictor : {ml_status}")
    log.info(f"  Chat      : {llm_status}")
    log.info(f"  URL       : http://localhost:5000")
    log.info("=" * 58)

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)