"""
NJN Neonatal Jaundice Detection — FastAPI Backend
Models: Hybrid Fusion (Final) + PCA-Hybrid (160)
Author: Md Abu Sayem | MRIIRS
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import cv2
import joblib
import json
import warnings
import os
from scipy.stats import skew, kurtosis
import io

warnings.filterwarnings("ignore")

# ── TensorFlow / Keras ──────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.get_logger().setLevel('ERROR')

# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════
app = FastAPI(
    title="NJN Jaundice Detection API",
    description="Non-invasive neonatal jaundice screening using multi-region hybrid fusion",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# LOAD MODELS AT STARTUP
# ═══════════════════════════════════════════════════════════════
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "NJN_EffNet_Full_vs_PCAHybrid_V3_bundle.joblib")
KERAS_PATH  = os.path.join(BASE_DIR, "NJN_EffNetB0_feature_extractor_V3.keras")
META_PATH   = os.path.join(BASE_DIR, "NJN_EffNet_Full_vs_PCAHybrid_V3_deployment_meta.json")

print("Loading bundle...")
bundle       = joblib.load(BUNDLE_PATH)
imputer      = bundle["imputer"]
pca_model    = bundle["pca_model"]
svm_full     = bundle["svm_full"]
svm_pca      = bundle["svm_pca"]
classical_cols = bundle["classical_feature_columns"]
pca_dim      = bundle["pca_dim"]
res_full     = bundle["results_full"]
res_pca      = bundle["results_pca"]

THRESHOLD_FULL = res_full["youden_thr"]
THRESHOLD_PCA  = res_pca["youden_thr"]

print("Loading EfficientNetB0...")
eff = tf.keras.models.load_model(KERAS_PATH)
eff.trainable = False

with open(META_PATH) as f:
    meta = json.load(f)

IMG_SIZE = 224
print("✅ All models loaded successfully!")

# ═══════════════════════════════════════════════════════════════
# ROI EXTRACTION
# ═══════════════════════════════════════════════════════════════
def get_forehead_box_classical(img_bgr):
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  20,  40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = np.ones((5, 5), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    min_area   = max(300, int(0.002 * h * w))
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area >= min_area:
            candidates.append((i, x, y, bw, bh, area))
    if candidates:
        candidates = sorted(candidates, key=lambda t: (t[2], -t[5]))
        _, x, y, bw, bh, _ = candidates[0]
        pad_x = int(0.05 * bw); pad_y = int(0.05 * bh)
        x  = max(0, x - pad_x);   y  = max(0, y - pad_y)
        bw = min(w - x, bw + 2*pad_x); bh = min(h - y, bh + 2*pad_y)
        x1 = max(0, x + int(0.28*bw)); x2 = min(w, x + int(0.68*bw))
        y1 = max(0, y + int(0.05*bh)); y2 = min(h, y + int(0.15*bh))
    else:
        x1, x2 = int(0.34*w), int(0.60*w)
        y1, y2 = int(0.08*h), int(0.15*h)
    if (x2-x1) < 12 or (y2-y1) < 12:
        x1, x2 = int(0.34*w), int(0.60*w)
        y1, y2 = int(0.08*h), int(0.15*h)
    return x1, y1, x2, y2

def extract_forehead_roi_classical(img_bgr):
    x1,y1,x2,y2 = get_forehead_box_classical(img_bgr)
    roi = img_bgr[y1:y2, x1:x2]
    return roi if roi.size else None

def extract_chest_roi_classical(img_bgr):
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.50*h):int(0.72*h), int(0.28*w):int(0.78*w)]
    return roi if roi.size else None

def extract_abdomen_roi_classical(img_bgr):
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.60*h):int(0.92*h), int(0.25*w):int(0.75*w)]
    return roi if roi.size else None

def extract_forehead_roi_deep(img_bgr):
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.10*h):int(0.40*h), int(0.25*w):int(0.75*w)]
    return roi if roi.size else None

def extract_chest_roi_deep(img_bgr):
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.40*h):int(0.80*h), int(0.25*w):int(0.75*w)]
    return roi if roi.size else None

def extract_abdomen_roi_deep(img_bgr):
    h, w = img_bgr.shape[:2]
    roi = img_bgr[int(0.55*h):int(0.92*h), int(0.25*w):int(0.75*w)]
    return roi if roi.size else None

# ═══════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def gray_world_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    m = (mb + mg + mr) / 3.0
    b = b * (m / (mb + 1e-6))
    g = g * (m / (mg + 1e-6))
    r = r * (m / (mr + 1e-6))
    return np.clip(cv2.merge([b,g,r]), 0, 255).astype(np.uint8)

def normalize_clahe_lab(roi_bgr, clip=2.5):
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    L2    = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2,a,b]), cv2.COLOR_LAB2BGR)

def preprocess_roi(roi_bgr):
    roi_bgr = gray_world_white_balance(roi_bgr)
    roi_bgr = normalize_clahe_lab(roi_bgr, clip=2.5)
    return roi_bgr

def skin_mask_hsv(roi_bgr):
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0,  20,  40], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    k     = np.ones((3,3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
def masked_stats(channel, mask):
    vals = channel[mask == 255].astype(np.float32)
    if vals.size < 30:
        return (np.nan,) * 6
    return (
        float(np.mean(vals)), float(np.std(vals)),
        float(np.median(vals)), float(skew(vals)),
        float(kurtosis(vals)), float(np.percentile(vals, 90))
    )

def extract_classical_features(roi_bgr):
    roi_bgr = preprocess_roi(roi_bgr)
    mask    = skin_mask_hsv(roi_bgr)
    if np.sum(mask == 255) < 200:
        mask = np.ones(roi_bgr.shape[:2], dtype=np.uint8) * 255

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = roi_rgb[:,:,0], roi_rgb[:,:,1], roi_rgb[:,:,2]

    Rm,Rs,_,_,_,R90 = masked_stats(R, mask)
    Gm,Gs,_,_,_,G90 = masked_stats(G, mask)
    Bm,Bs,_,_,_,B90 = masked_stats(B, mask)

    ygi       = float((Rm + Gm) / (Bm + 1e-6))
    r_over_b  = float(Rm / (Bm + 1e-6))
    g_over_b  = float(Gm / (Bm + 1e-6))
    rg_balance= float((Rm - Gm) / (Rm + Gm + 1e-6))
    denom     = Rm + Gm + Bm + 1e-6
    r_ch, g_ch, b_ch = float(Rm/denom), float(Gm/denom), float(Bm/denom)

    hsv = cv2.cvtColor(roi_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    Hm,_,_,_,_,_ = masked_stats(H, mask)
    Sm,_,_,_,_,_ = masked_stats(S, mask)
    Vm,Vs,_,_,_,_ = masked_stats(V, mask)
    Hvals     = H[mask == 255]
    hue_peak  = float(np.argmax(np.histogram(Hvals, bins=18, range=(0,180))[0])) \
                if Hvals.size > 30 else np.nan

    lab = cv2.cvtColor(roi_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    Lc, ac, bc = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    Lm,_,_,_,_,_ = masked_stats(Lc, mask)
    am,_,_,_,_,_ = masked_stats(ac, mask)
    bm,_,_,_,_,b90 = masked_stats(bc, mask)
    bvals       = bc[mask == 255]
    yellow_prop = float(np.mean(bvals > np.percentile(bvals, 75))) \
                  if bvals.size > 30 else np.nan

    feats = {
        "R_mean":Rm,"G_mean":Gm,"B_mean":Bm,
        "R_std":Rs,"G_std":Gs,"B_std":Bs,
        "R_p90":R90,"G_p90":G90,"B_p90":B90,
        "YGI":ygi,"R_over_B":r_over_b,"G_over_B":g_over_b,
        "RG_balance":rg_balance,
        "r_ch":r_ch,"g_ch":g_ch,"b_ch":b_ch,
        "H_mean":Hm,"S_mean":Sm,"V_mean":Vm,"V_std":Vs,
        "Hue_peak_bin":hue_peak,
        "L_mean":Lm,"a_mean":am,"Lab_b_mean":bm,"Lab_b_p90":b90,
        "yellow_prop":yellow_prop,
    }
    feats["YGI_x_b"] = feats["YGI"] * feats["Lab_b_mean"]
    return feats

def roi_to_tensor(roi_bgr):
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    x   = roi.astype(np.float32)
    return preprocess_input(x)

def get_eff_embedding(img_bgr, roi_fn):
    roi = roi_fn(img_bgr)
    if roi is None:
        return None
    x   = roi_to_tensor(roi)
    emb = eff(np.expand_dims(x, 0), training=False).numpy().ravel()
    return emb

# ═══════════════════════════════════════════════════════════════
# CORE PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════
def predict_from_bytes(image_bytes: bytes, mode: str = "pca") -> dict:
    """
    mode: 'pca'  → PCA-Hybrid (160)  [recommended, fast]
          'full' → Hybrid Fusion (Final) [best accuracy]
          'both' → returns both
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Send a valid JPEG/PNG.")

    # Classical features
    fh_c = extract_forehead_roi_classical(img)
    ch_c = extract_chest_roi_classical(img)
    ab_c = extract_abdomen_roi_classical(img)
    if fh_c is None or ch_c is None or ab_c is None:
        raise ValueError("ROI extraction failed. Ensure image shows a newborn clearly.")

    fh_feat = extract_classical_features(fh_c)
    ch_feat = extract_classical_features(ch_c)
    ab_feat = extract_classical_features(ab_c)

    row = {}
    row.update({f"FH_{k}": v for k,v in fh_feat.items()})
    row.update({f"CH_{k}": v for k,v in ch_feat.items()})
    row.update({f"AB_{k}": v for k,v in ab_feat.items()})

    xc_df = pd.DataFrame([row])[classical_cols]
    xc    = imputer.transform(xc_df)

    # Deep embeddings
    e_fh = get_eff_embedding(img, extract_forehead_roi_deep)
    e_ch = get_eff_embedding(img, extract_chest_roi_deep)
    e_ab = get_eff_embedding(img, extract_abdomen_roi_deep)
    if e_fh is None or e_ch is None or e_ab is None:
        raise ValueError("Deep embedding extraction failed.")

    deep = np.hstack([e_fh, e_ch, e_ab]).reshape(1, -1)

    results = {}

    if mode in ("pca", "both"):
        deep_pca = pca_model.transform(deep)
        X_pca    = np.hstack([xc, deep_pca])
        prob_pca = float(svm_pca.predict_proba(X_pca)[:, 1][0])
        pred_pca = int(prob_pca >= THRESHOLD_PCA)
        results["pca"] = {
            "model":               "PCA-Hybrid (160)",
            "probability_jaundice": round(prob_pca, 4),
            "threshold":           round(THRESHOLD_PCA, 4),
            "predicted_label":     pred_pca,
            "predicted_class":     "Jaundice" if pred_pca == 1 else "Normal",
            "confidence_pct":      round(prob_pca * 100, 1),
            "risk_level":          _risk_level(prob_pca),
        }

    if mode in ("full", "both"):
        X_full    = np.hstack([xc, deep])
        prob_full = float(svm_full.predict_proba(X_full)[:, 1][0])
        pred_full = int(prob_full >= THRESHOLD_FULL)
        results["full"] = {
            "model":               "Hybrid Fusion (Final)",
            "probability_jaundice": round(prob_full, 4),
            "threshold":           round(THRESHOLD_FULL, 4),
            "predicted_label":     pred_full,
            "predicted_class":     "Jaundice" if pred_full == 1 else "Normal",
            "confidence_pct":      round(prob_full * 100, 1),
            "risk_level":          _risk_level(prob_full),
        }

    return results

def _risk_level(prob: float) -> str:
    if prob >= 0.75:   return "HIGH"
    elif prob >= 0.45: return "MODERATE"
    elif prob >= 0.25: return "LOW"
    else:              return "VERY_LOW"

def _advice(predicted_class: str, risk_level: str) -> dict:
    if predicted_class == "Jaundice":
        if risk_level == "HIGH":
            return {
                "summary": "High jaundice risk detected. Seek medical attention immediately.",
                "steps": [
                    "Consult a paediatrician or visit the nearest hospital immediately.",
                    "Do not delay — high bilirubin can cause brain damage if untreated.",
                    "A blood test (TSB) is required to confirm the diagnosis.",
                    "Keep the baby well-fed; frequent feeding helps reduce bilirubin.",
                    "Do not expose the baby to direct sunlight as a substitute for treatment."
                ],
                "urgency": "IMMEDIATE"
            }
        else:
            return {
                "summary": "Moderate jaundice risk detected. Medical review is recommended.",
                "steps": [
                    "Schedule a paediatrician visit within 24 hours.",
                    "Monitor the baby's skin colour — yellowness spreading to the chest or abdomen needs urgent care.",
                    "Ensure frequent breastfeeding (8–12 times per day).",
                    "A transcutaneous or serum bilirubin test is advised for confirmation.",
                    "Watch for signs of lethargy, poor feeding, or high-pitched crying."
                ],
                "urgency": "SOON"
            }
    else:
        return {
            "summary": "No significant jaundice detected at this time.",
            "steps": [
                "Continue regular monitoring of skin colour over the next few days.",
                "Ensure adequate feeding — breast milk or formula.",
                "If yellowing appears or worsens, repeat the screening or consult a doctor.",
                "Normal newborns may develop mild jaundice in the first week; this usually resolves on its own.",
                "Follow your regular postnatal care schedule."
            ],
            "urgency": "ROUTINE"
        }

# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {
        "name":    "NJN Jaundice Detection API",
        "version": "3.0.0",
        "status":  "running",
        "models":  ["PCA-Hybrid (160)", "Hybrid Fusion (Final)"],
        "endpoints": {
            "predict_pca":  "POST /predict",
            "predict_full": "POST /predict?mode=full",
            "predict_both": "POST /predict?mode=both",
            "health":       "GET /health",
            "meta":         "GET /meta"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": True}

@app.get("/meta")
async def get_meta():
    return meta

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = "pca"           # "pca" | "full" | "both"
):
    """
    POST /predict?mode=pca
    Body: multipart/form-data with field 'file' (JPEG or PNG image)
    Returns: prediction result with probability, class, risk level, and advice
    """
    if mode not in ("pca", "full", "both"):
        raise HTTPException(status_code=400, detail="mode must be 'pca', 'full', or 'both'")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg",
                                   "image/webp", "image/bmp"):
        raise HTTPException(status_code=400,
                            detail="Only JPEG, PNG, WEBP, BMP images are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image size must be under 10MB.")

    try:
        results = predict_from_bytes(image_bytes, mode=mode)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Pick primary result for advice
    primary = results.get("pca") or results.get("full")
    advice  = _advice(primary["predicted_class"], primary["risk_level"])

    return JSONResponse({
        "success":      True,
        "predictions":  results,
        "primary_model":"PCA-Hybrid (160)" if "pca" in results else "Hybrid Fusion (Final)",
        "advice":       advice,
        "disclaimer":   "This is an AI-based screening tool. Not a medical diagnosis. Always consult a qualified paediatrician."
    })
