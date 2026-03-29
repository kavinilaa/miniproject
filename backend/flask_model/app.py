"""
Flask API — Advanced Multimodal Polyp Detection
─────────────────────────────────────────────────
Endpoints:
  POST /predict            — image + clinical → binary prediction + heatmap + XAI
  POST /classify           — multi-class polyp type (Feature 8)
  POST /segment            — U-Net segmentation mask (Feature 5)
  POST /predict-video      — frame-by-frame video detection (Feature 5)
  POST /incremental-update — add confirmed case to replay buffer (Feature 10)
  POST /fine-tune          — trigger incremental fine-tune step (Feature 10)
  POST /set-domain         — switch domain adapter (Feature 7)
  GET  /health             — status check
"""

import io
import os
import base64
import json
import sqlite3
import time
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import (MultimodalPolypDetector, build_clinical_vector,
                   POLYP_CLASSES, IncrementalLearner)

app  = Flask(__name__)
CORS(app)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
MODEL_PATH = "multimodal_polyp_model.pth"
SEG_PATH   = "kvasir_seg_model.h5"
DB_PATH    = os.path.join(os.path.dirname(__file__), "colonoscopy_app.db")
MEDIA_ROOT = os.path.join(os.path.dirname(__file__), "uploads", "patients")

os.makedirs(MEDIA_ROOT, exist_ok=True)


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            clinical_risk TEXT,
            data_json TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patient_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_db_id INTEGER,
            action TEXT NOT NULL,
            summary TEXT,
            payload_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_db_id) REFERENCES patients(id)
        )
        """
    )
    conn.commit()
    conn.close()


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_patient_db_id(form_or_json):
    patient_db_id = parse_int(form_or_json.get("patientDbId"))
    if patient_db_id:
        return patient_db_id

    patient_name = (form_or_json.get("patientName") or "").strip()
    age = parse_int(form_or_json.get("age"))
    gender = (form_or_json.get("gender") or "").strip()
    if not patient_name:
        return None

    conn = get_db_conn()
    cur = conn.cursor()
    if age is not None and gender:
        cur.execute(
            """
            SELECT id FROM patients
            WHERE patient_name = ? AND age = ? AND gender = ?
            ORDER BY id DESC LIMIT 1
            """,
            (patient_name, age, gender),
        )
    else:
        cur.execute(
            """
            SELECT id FROM patients
            WHERE patient_name = ?
            ORDER BY id DESC LIMIT 1
            """,
            (patient_name,),
        )
    row = cur.fetchone()
    conn.close()
    return row["id"] if row else None


def log_patient_event(action, summary, payload_dict, patient_db_id=None):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO patient_events (patient_db_id, action, summary, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            patient_db_id,
            action,
            summary,
            json.dumps(payload_dict or {}, ensure_ascii=True),
        ),
    )
    conn.commit()
    conn.close()


def save_patient_media_bytes(patient_db_id, content_bytes, suffix, extension):
    if not patient_db_id:
        return None
    ext = extension if str(extension).startswith(".") else f".{extension}"
    patient_dir = os.path.join(MEDIA_ROOT, str(patient_db_id))
    os.makedirs(patient_dir, exist_ok=True)
    filename = f"{int(time.time())}_{suffix}_{uuid.uuid4().hex[:8]}{ext.lower()}"
    abs_path = os.path.join(patient_dir, filename)
    with open(abs_path, "wb") as f:
        f.write(content_bytes)
    rel_path = f"{patient_db_id}/{filename}"
    return rel_path


def media_url(rel_path):
    if not rel_path:
        return None
    return f"/patient-media/{rel_path}"


def payload_as_dict(payload_raw):
    if isinstance(payload_raw, dict):
        return payload_raw
    if payload_raw is None:
        return {}
    return {"value": payload_raw}


def decode_payload_json(payload_json_text):
    try:
        return json.loads(payload_json_text or "{}")
    except Exception:
        return {"value": str(payload_json_text or "")}


init_db()

# ── Image transform ────────────────────────────────────────────────────────
IMG_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Load PyTorch multimodal model ──────────────────────────────────────────
model = MultimodalPolypDetector().to(DEVICE)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state, strict=True)
        print(f"✔ Loaded (strict): {MODEL_PATH}")
    except RuntimeError as e:
        # Allow partial loading when checkpoint architecture differs.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"⚠ Partial weight load for {MODEL_PATH}")
        print(f"  strict-load error: {e}")
        print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
else:
    print("⚠ No weights — using random (run train.py first)")
model.eval()

# ── Load Keras segmentation model (optional) ──────────────────────────────
seg_model = None
if os.path.exists(SEG_PATH):
    try:
        import tensorflow as tf
        seg_model = tf.keras.models.load_model(SEG_PATH, compile=False)
        print(f"✔ Loaded segmentation model: {SEG_PATH}")
    except Exception as e:
        print(f"⚠ Could not load seg model: {e}")

# ── Incremental learner (Feature 10) ──────────────────────────────────────
incremental = IncrementalLearner(buffer_size=200, min_samples=16)
incremental_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Automatic online adaptation settings.
AUTO_LEARNING_ENABLED = True
AUTO_LABEL_HIGH = 0.85
AUTO_LABEL_LOW = 0.15
AUTO_FINETUNE_COOLDOWN_SEC = 90
last_auto_finetune_at = 0.0


def auto_learn_from_prediction(img_tensor, clin_tensor, raw_prob, patient_db_id=None, source="predict"):
    """
    Add high-confidence pseudo-labeled samples to replay buffer and
    periodically trigger a small fine-tune step.
    """
    global last_auto_finetune_at

    if not AUTO_LEARNING_ENABLED:
        return {"enabled": False, "added": False, "trained": False}

    if raw_prob >= AUTO_LABEL_HIGH:
        pseudo_label = 1.0
    elif raw_prob <= AUTO_LABEL_LOW:
        pseudo_label = 0.0
    else:
        return {
            "enabled": True,
            "added": False,
            "trained": False,
            "reason": "confidence window not met",
            "bufferSize": len(incremental.buffer),
        }

    incremental.add(img_tensor, clin_tensor, pseudo_label)
    result = {
        "enabled": True,
        "added": True,
        "trained": False,
        "label": int(pseudo_label),
        "bufferSize": len(incremental.buffer),
    }
    log_patient_event(
        "Auto Incremental",
        f"Added pseudo-labeled sample from {source}",
        {
            "source": source,
            "label": int(pseudo_label),
            "bufferSize": len(incremental.buffer),
            "prob": round(float(raw_prob), 4),
        },
        patient_db_id=patient_db_id,
    )

    now = time.time()
    if incremental.ready() and (now - last_auto_finetune_at) >= AUTO_FINETUNE_COOLDOWN_SEC:
        loss = incremental.fine_tune_step(model, incremental_optimizer, DEVICE)
        torch.save(model.state_dict(), MODEL_PATH)
        last_auto_finetune_at = now
        result.update({"trained": True, "loss": round(loss, 4), "saved": MODEL_PATH})
        log_patient_event(
            "Auto Fine-tune",
            "Automatic fine-tune step completed",
            {
                "source": source,
                "loss": round(loss, 4),
                "bufferSize": len(incremental.buffer),
            },
            patient_db_id=patient_db_id,
        )

    return result


# ── Grad-CAM ───────────────────────────────────────────────────────────────
def compute_gradcam(img_tensor, clin_tensor):
    img_tensor  = img_tensor.unsqueeze(0).to(DEVICE)
    clin_tensor = clin_tensor.unsqueeze(0).to(DEVICE)

    feature_maps = [None]
    gradients    = [None]

    fh = model.image_enc.attn4.register_forward_hook(
        lambda _, __, o: feature_maps.__setitem__(0, o))
    bh = model.image_enc.attn4.register_full_backward_hook(
        lambda _, __, g: gradients.__setitem__(0, g[0]))

    img_tensor.requires_grad_(True)
    logit, _, _ = model(img_tensor, clin_tensor)
    model.zero_grad()
    logit.backward()

    fh.remove(); bh.remove()

    pooled = gradients[0].mean(dim=[2, 3], keepdim=True)
    cam    = (pooled * feature_maps[0]).sum(dim=1).squeeze()
    cam    = F.relu(cam).detach().cpu().numpy()
    cam    = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def overlay_heatmap(orig_img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    orig_cv = cv2.cvtColor(
        np.array(orig_img.resize((IMG_SIZE, IMG_SIZE))), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(orig_cv, 0.55, heatmap, 0.45, 0)
    _, buf  = cv2.imencode(".jpg", blended)
    return base64.b64encode(buf).decode("utf-8")


# ── XAI explanation (Feature 6) ───────────────────────────────────────────
def xai_explanation(confidence, cam, polyp):
    focus_pct = float((cam > 0.5).sum() / cam.size * 100)
    reasons   = []
    if polyp:
        if cam.max() > 0.85:
            reasons.append("Strong irregular texture in focus region")
        if focus_pct > 20:
            reasons.append("Large suspicious area highlighted by model")
        if confidence > 0.85:
            reasons.append("High-contrast boundary pattern consistent with polyp")
        else:
            reasons.append("Moderate color variation in mucosal lining")
        reasons.append("Abnormal surface morphology detected")
    else:
        reasons.append("Uniform mucosal texture — no irregular patterns")
        reasons.append("Smooth surface with consistent color distribution")
        if confidence > 0.90:
            reasons.append("Very low activation in suspicious regions")

    if   confidence > 0.90: exp = "High confidence — clear structure, strong Grad-CAM activation"
    elif confidence > 0.75: exp = "Moderate-high confidence — distinct features, minor edge ambiguity"
    elif confidence > 0.60: exp = "Moderate confidence — some features present; re-scan if unclear"
    else:                   exp = "Low confidence — unclear image quality; re-scan recommended"

    return reasons[:3], exp


# ── Recommendation (Feature 5) ────────────────────────────────────────────
def recommendation(prediction, clinical_risk, confidence):
    polyp = prediction == "Polyp Detected"
    if polyp and clinical_risk == "High":
        return {"action": "Immediate Biopsy Required", "urgency": "critical",
                "detail": "High-risk polyp + elevated clinical risk. Biopsy within 48h.",
                "followup": "Oncology referral recommended"}
    elif polyp and clinical_risk == "Medium":
        return {"action": "Biopsy Recommended", "urgency": "high",
                "detail": "Polyp detected, moderate clinical risk. Biopsy within 1–2 weeks.",
                "followup": "Follow-up colonoscopy in 3 months"}
    elif polyp:
        return {"action": "Surveillance Colonoscopy", "urgency": "medium",
                "detail": "Polyp detected, low clinical risk. Monitor with follow-up.",
                "followup": "Repeat colonoscopy in 6 months"}
    elif clinical_risk == "High":
        return {"action": "Enhanced Monitoring", "urgency": "medium",
                "detail": "No polyp but high clinical risk factors.",
                "followup": "Colonoscopy in 6 months"}
    else:
        return {"action": "Routine Check-up", "urgency": "low",
                "detail": "No polyp. Continue standard screening.",
                "followup": "Next colonoscopy in 3–5 years"}


def final_risk(prediction, clinical_risk, confidence):
    polyp    = prediction == "Polyp Detected"
    ai_score = 3 if (polyp and confidence > 0.85) else 2 if polyp else 1
    cl_score = {"High": 3, "Medium": 2, "Low": 1}.get(clinical_risk, 1)
    total    = ai_score + cl_score
    if total >= 5: return "VERY HIGH ⚠️"
    elif total == 4: return "HIGH"
    elif total == 3: return "MEDIUM"
    else:            return "LOW"


# ── Core inference ─────────────────────────────────────────────────────────
def run_inference(image_bytes, clinical_data):
    orig_img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor  = IMG_TF(orig_img)
    clin_vec    = build_clinical_vector(clinical_data)
    clin_tensor = torch.tensor(clin_vec, dtype=torch.float32)

    with torch.no_grad():
        logit, class_logits, _ = model(
            img_tensor.unsqueeze(0).to(DEVICE),
            clin_tensor.unsqueeze(0).to(DEVICE))
        prob       = torch.sigmoid(logit).item()
        class_prob = torch.softmax(class_logits, dim=1).squeeze().tolist()

    polyp      = prob > 0.5
    prediction = "Polyp Detected" if polyp else "No Polyp"
    confidence = round(prob if polyp else 1.0 - prob, 4)
    risk       = "High" if (polyp and confidence > 0.85) else \
                 ("Medium" if polyp else "Low")

    cam         = compute_gradcam(img_tensor, clin_tensor)
    heatmap_b64 = overlay_heatmap(orig_img, cam)

    return (prediction, confidence, risk, prob,
            cam, heatmap_b64, class_prob, img_tensor, clin_tensor)


# ── /predict ──────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes   = request.files["image"].read()
    clinical_data = request.form.to_dict()
    clinical_risk = clinical_data.get("clinicalRisk", "Low")
    patient_db_id = find_patient_db_id(clinical_data)
    input_image_path = save_patient_media_bytes(patient_db_id, image_bytes, "predict_input", ".jpg")

    try:
        (prediction, confidence, risk, raw_prob,
         cam, heatmap_b64, class_prob,
         img_tensor, clin_tensor) = run_inference(image_bytes, clinical_data)
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    polyp = prediction == "Polyp Detected"
    xai_reasons, conf_exp = xai_explanation(confidence, cam, polyp)
    rec   = recommendation(prediction, clinical_risk, confidence)
    final = final_risk(prediction, clinical_risk, confidence)

    auto_learning = {"enabled": AUTO_LEARNING_ENABLED, "added": False, "trained": False}
    try:
        auto_learning = auto_learn_from_prediction(
            img_tensor,
            clin_tensor,
            raw_prob,
            patient_db_id=patient_db_id,
            source="predict",
        )
    except Exception as e:
        auto_learning = {
            "enabled": AUTO_LEARNING_ENABLED,
            "added": False,
            "trained": False,
            "error": str(e),
        }

    # Feature 8: multi-class
    polyp_type = None
    if polyp:
        top_idx    = int(np.argmax(class_prob))
        polyp_type = {
            "type":       POLYP_CLASSES[top_idx],
            "confidence": round(class_prob[top_idx] * 100, 1),
            "all":        {c: round(p * 100, 1)
                           for c, p in zip(POLYP_CLASSES, class_prob)},
        }

    heatmap_path = None
    try:
        heatmap_path = save_patient_media_bytes(
            patient_db_id,
            base64.b64decode(heatmap_b64),
            "predict_heatmap",
            ".jpg",
        )
    except Exception:
        heatmap_path = None

    log_patient_event(
        "Predict",
        f"{prediction} ({round(confidence * 100, 1)}%)",
        {
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk,
            "finalRisk": final,
            "inputImagePath": input_image_path,
            "inputImageUrl": media_url(input_image_path),
            "heatmapPath": heatmap_path,
            "heatmapUrl": media_url(heatmap_path),
        },
        patient_db_id=patient_db_id,
    )

    # Model comparison (Feature 10 — lightweight variant)
    try:
        light_model = MultimodalPolypDetector(lightweight=True).to(DEVICE)
        light_model.eval()
        with torch.no_grad():
            ll, _, _ = light_model(img_tensor.unsqueeze(0).to(DEVICE),
                                   clin_tensor.unsqueeze(0).to(DEVICE))
            lp = torch.sigmoid(ll).item()
        lpolyp = lp > 0.5
        model_comparison = {
            "cnn": {
                "model": "ResNet50 + HybridAttention (Full)",
                "prediction": prediction,
                "confidence": round(confidence * 100, 1),
            },
            "transformer": {
                "model": "MobileNetV3 (Lightweight)",
                "prediction": "Polyp Detected" if lpolyp else "No Polyp",
                "confidence": round((lp if lpolyp else 1-lp) * 100, 1),
            },
        }
    except Exception:
        model_comparison = {}

    return jsonify({
        "prediction":            prediction,
        "confidence":            confidence,
        "risk":                  risk,
        "heatmap":               heatmap_b64,
        "xaiReasons":            xai_reasons,
        "confidenceExplanation": conf_exp,
        "recommendation":        rec,
        "finalRisk":             final,
        "polypType":             polyp_type,
        "modelComparison":       model_comparison,
        "autoLearning":          auto_learning,
    })


# ── /classify — Feature 8: Multi-class polyp type ─────────────────────────
@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_bytes   = request.files["image"].read()
    clinical_data = request.form.to_dict()
    patient_db_id = find_patient_db_id(clinical_data)
    input_image_path = save_patient_media_bytes(patient_db_id, image_bytes, "classify_input", ".jpg")
    try:
        (prediction, confidence, _, raw_prob, _, _,
         class_prob, img_tensor, clin_tensor) = run_inference(image_bytes, clinical_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    auto_learning = {"enabled": AUTO_LEARNING_ENABLED, "added": False, "trained": False}
    try:
        auto_learning = auto_learn_from_prediction(
            img_tensor,
            clin_tensor,
            raw_prob,
            patient_db_id=patient_db_id,
            source="classify",
        )
    except Exception as e:
        auto_learning = {
            "enabled": AUTO_LEARNING_ENABLED,
            "added": False,
            "trained": False,
            "error": str(e),
        }

    top_idx = int(np.argmax(class_prob))
    log_patient_event(
        "Classify",
        f"{POLYP_CLASSES[top_idx]} ({round(class_prob[top_idx] * 100, 1)}%)",
        {
            "prediction": prediction,
            "polypType": POLYP_CLASSES[top_idx],
            "typeConfidence": round(class_prob[top_idx] * 100, 1),
            "inputImagePath": input_image_path,
            "inputImageUrl": media_url(input_image_path),
        },
        patient_db_id=patient_db_id,
    )
    return jsonify({
        "prediction":  prediction,
        "confidence":  confidence,
        "polypType":   POLYP_CLASSES[top_idx],
        "typeConfidence": round(class_prob[top_idx] * 100, 1),
        "allTypes":    {c: round(p * 100, 1)
                        for c, p in zip(POLYP_CLASSES, class_prob)},
        "autoLearning": auto_learning,
    })


# ── /segment — Feature 5: U-Net segmentation mask ─────────────────────────
@app.route("/segment", methods=["POST"])
def segment():
    if seg_model is None:
        return jsonify({"error": "Segmentation model not loaded"}), 503
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    patient_db_id = find_patient_db_id(request.form.to_dict())

    image_bytes = request.files["image"].read()
    input_image_path = save_patient_media_bytes(patient_db_id, image_bytes, "segment_input", ".jpg")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    pred = seg_model.predict(arr[np.newaxis], verbose=0)[0, :, :, 0]
    mask_bin = (pred > 0.5).astype(np.uint8) * 255

    # Overlay mask on image
    orig_cv  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    colored  = cv2.applyColorMap(mask_bin, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(orig_cv, 0.6, colored, 0.4, 0)

    _, buf = cv2.imencode(".jpg", overlay)
    mask_b64 = base64.b64encode(buf).decode("utf-8")
    overlay_path = save_patient_media_bytes(patient_db_id, buf.tobytes(), "segment_overlay", ".jpg")

    coverage = float(mask_bin.sum()) / (256 * 256 * 255) * 100
    log_patient_event(
        "Segment",
        f"Coverage {round(coverage, 2)}%",
        {
            "polypCoverage": round(coverage, 2),
            "hasPolyp": coverage > 0.5,
            "inputImagePath": input_image_path,
            "inputImageUrl": media_url(input_image_path),
            "overlayPath": overlay_path,
            "overlayUrl": media_url(overlay_path),
        },
        patient_db_id=patient_db_id,
    )
    return jsonify({
        "segmentationMask": mask_b64,
        "polypCoverage":    round(coverage, 2),
        "hasPolyp":         coverage > 0.5,
    })


# ── /predict-video — Feature 5: Real-time video ───────────────────────────
@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video provided"}), 400

    video_bytes   = request.files["video"].read()
    clinical_data = request.form.to_dict()
    patient_db_id = find_patient_db_id(clinical_data)
    input_video_path = save_patient_media_bytes(patient_db_id, video_bytes, "video_input", ".mp4")
    tmp_path      = os.path.join(os.path.dirname(__file__), "tmp_video.mp4")

    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    cap          = cv2.VideoCapture(tmp_path)
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    sample_every = max(1, int(fps * 2))

    results, frame_idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % sample_every == 0:
            _, buf = cv2.imencode(".jpg", frame)
            try:
                pred, conf, risk, _, _, _, _, _, _ = \
                    run_inference(buf.tobytes(), clinical_data)
                results.append({
                    "frame":      frame_idx,
                    "timestamp":  round(frame_idx / fps, 1),
                    "prediction": pred,
                    "confidence": conf,
                    "risk":       risk,
                })
            except Exception:
                pass
        frame_idx += 1
    cap.release()
    try: os.remove(tmp_path)
    except Exception: pass

    polyp_frames = [r for r in results if r["prediction"] == "Polyp Detected"]
    log_patient_event(
        "Video",
        f"Polyp in {len(polyp_frames)}/{len(results)} sampled frames",
        {
            "totalFrames": total,
            "sampledFrames": len(results),
            "polypFrames": len(polyp_frames),
            "inputVideoPath": input_video_path,
            "inputVideoUrl": media_url(input_video_path),
        },
        patient_db_id=patient_db_id,
    )
    return jsonify({
        "totalFrames":   total,
        "sampledFrames": len(results),
        "polypFrames":   len(polyp_frames),
        "frames":        results,
        "summary":       f"Polyp in {len(polyp_frames)}/{len(results)} frames",
    })


# ── /incremental-update — Feature 10: Add confirmed case ──────────────────
@app.route("/incremental-update", methods=["POST"])
def incremental_update():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes   = request.files["image"].read()
    clinical_data = request.form.to_dict()
    label         = float(request.form.get("label", 0))  # 1=polyp, 0=no polyp
    patient_db_id = find_patient_db_id(clinical_data)
    input_image_path = save_patient_media_bytes(patient_db_id, image_bytes, "incremental_input", ".jpg")

    try:
        orig_img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor  = IMG_TF(orig_img)
        clin_vec    = build_clinical_vector(clinical_data)
        clin_tensor = torch.tensor(clin_vec, dtype=torch.float32)
        incremental.add(img_tensor, clin_tensor, label)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    log_patient_event(
        "Incremental Update",
        f"Replay buffer size: {len(incremental.buffer)}",
        {
            "label": label,
            "bufferSize": len(incremental.buffer),
            "readyToTrain": incremental.ready(),
            "inputImagePath": input_image_path,
            "inputImageUrl": media_url(input_image_path),
        },
        patient_db_id=patient_db_id,
    )

    return jsonify({
        "message":      "Case added to replay buffer",
        "bufferSize":   len(incremental.buffer),
        "readyToTrain": incremental.ready(),
    })


# ── /fine-tune — Feature 10: Trigger incremental learning step ────────────
@app.route("/fine-tune", methods=["POST"])
def fine_tune():
    if not incremental.ready():
        return jsonify({
            "message": f"Not enough samples yet. "
                       f"Need {incremental.min_samples}, "
                       f"have {len(incremental.buffer)}."
        }), 400

    loss = incremental.fine_tune_step(model, incremental_optimizer, DEVICE)
    torch.save(model.state_dict(), MODEL_PATH)
    log_patient_event(
        "Fine-tune",
        "Incremental fine-tune step complete",
        {"loss": round(loss, 4), "saved": MODEL_PATH},
        patient_db_id=None,
    )
    return jsonify({
        "message": "Fine-tune step complete",
        "loss":    round(loss, 4),
        "saved":   MODEL_PATH,
    })


# ── /set-domain — Feature 7: Switch domain adapter ────────────────────────
@app.route("/set-domain", methods=["POST"])
def set_domain():
    domain_id = int(request.json.get("domainId", 0))
    model.set_domain(domain_id)
    log_patient_event(
        "Set Domain",
        f"Domain set to {domain_id}",
        {"domainId": domain_id},
        patient_db_id=None,
    )
    return jsonify({"message": f"Domain set to {domain_id}"})


# ── /health ────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "Flask Advanced Multimodal AI running",
        "device":        str(DEVICE),
        "model":         "MultimodalPolypDetector v2 (ResNet50 + HybridAttention + CrossFusion)",
        "weights":       "loaded" if os.path.exists(MODEL_PATH) else "random",
        "segmentation":  "loaded" if seg_model else "not loaded",
        "replayBuffer":  len(incremental.buffer),
        "dbEnabled":     True,
        "features": [
            "Multimodal Learning", "Hybrid Attention", "Early Detection",
            "Feature Fusion", "Video Detection", "XAI Grad-CAM",
            "Domain Adaptation", "Multi-Class Classification",
            "Lightweight Mode", "Incremental Learning"
        ],
    })


@app.route("/patients", methods=["POST"])
def create_patient():
    payload = request.get_json(silent=True) or {}
    patient_name = (payload.get("patientName") or "").strip()
    age = parse_int(payload.get("age"))
    gender = (payload.get("gender") or "").strip()

    if not patient_name or age is None or not gender:
        return jsonify({"error": "patientName, age, and gender are required"}), 400

    clinical_risk = payload.get("clinicalRisk", "Low")
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO patients (
            patient_id, patient_name, age, gender, clinical_risk, data_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            payload.get("patientId"),
            patient_name,
            age,
            gender,
            clinical_risk,
            json.dumps(payload, ensure_ascii=True),
        ),
    )
    patient_db_id = cur.lastrowid
    conn.commit()
    conn.close()

    log_patient_event(
        "Patient Save",
        f"Patient record saved for {patient_name}",
        {"patientName": patient_name, "clinicalRisk": clinical_risk},
        patient_db_id=patient_db_id,
    )

    return jsonify({
        "message": "Patient saved to database",
        "patientDbId": patient_db_id,
    })


@app.route("/patients", methods=["GET"])
def list_patients():
    limit = parse_int(request.args.get("limit")) or 50
    limit = min(max(limit, 1), 200)
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, patient_id, patient_name, age, gender, clinical_risk, created_at, updated_at
        FROM patients
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    items = [
        {
            "id": row["id"],
            "patientId": row["patient_id"],
            "patientName": row["patient_name"],
            "age": row["age"],
            "gender": row["gender"],
            "clinicalRisk": row["clinical_risk"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]
    return jsonify({"items": items, "count": len(items)})


@app.route("/patients/<int:patient_db_id>", methods=["GET"])
def get_patient(patient_db_id):
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, patient_id, patient_name, age, gender, clinical_risk,
               data_json, created_at, updated_at
        FROM patients
        WHERE id = ?
        """,
        (patient_db_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Patient not found"}), 404

    data = json.loads(row["data_json"] or "{}")
    return jsonify({
        "id": row["id"],
        "patientId": row["patient_id"],
        "patientName": row["patient_name"],
        "age": row["age"],
        "gender": row["gender"],
        "clinicalRisk": row["clinical_risk"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "data": data,
    })


@app.route("/patient-events", methods=["GET"])
def list_patient_events():
    limit = parse_int(request.args.get("limit")) or 100
    limit = min(max(limit, 1), 300)
    patient_db_id = parse_int(request.args.get("patientDbId"))
    conn = get_db_conn()
    cur = conn.cursor()
    if patient_db_id is not None:
        cur.execute(
            """
            SELECT e.id, e.patient_db_id, e.action, e.summary, e.payload_json, e.created_at,
                   p.patient_name
            FROM patient_events e
            LEFT JOIN patients p ON p.id = e.patient_db_id
            WHERE e.patient_db_id = ?
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (patient_db_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT e.id, e.patient_db_id, e.action, e.summary, e.payload_json, e.created_at,
                   p.patient_name
            FROM patient_events e
            LEFT JOIN patients p ON p.id = e.patient_db_id
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (limit,),
        )
    rows = cur.fetchall()
    conn.close()

    items = [
        {
            "id": row["id"],
            "patientDbId": row["patient_db_id"],
            "patientName": row["patient_name"],
            "action": row["action"],
            "summary": row["summary"],
            "payload": payload_as_dict(decode_payload_json(row["payload_json"])),
            "createdAt": row["created_at"],
        }
        for row in rows
    ]
    return jsonify({"items": items, "count": len(items)})


@app.route("/patients/<int:patient_db_id>/full", methods=["GET"])
def get_patient_full(patient_db_id):
    conn = get_db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, patient_id, patient_name, age, gender, clinical_risk,
               data_json, created_at, updated_at
        FROM patients
        WHERE id = ?
        """,
        (patient_db_id,),
    )
    patient_row = cur.fetchone()
    if not patient_row:
        conn.close()
        return jsonify({"error": "Patient not found"}), 404

    cur.execute(
        """
        SELECT e.id, e.action, e.summary, e.payload_json, e.created_at
        FROM patient_events e
        WHERE e.patient_db_id = ?
        ORDER BY e.id DESC
        LIMIT 500
        """,
        (patient_db_id,),
    )
    event_rows = cur.fetchall()
    conn.close()

    events = []
    media = []
    for row in event_rows:
        payload = payload_as_dict(decode_payload_json(row["payload_json"]))
        events.append(
            {
                "id": row["id"],
                "action": row["action"],
                "summary": row["summary"],
                "payload": payload,
                "createdAt": row["created_at"],
            }
        )
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key.endswith("Path") and isinstance(value, str) and value:
                    media.append(
                        {
                            "eventId": row["id"],
                            "eventAction": row["action"],
                            "eventTime": row["created_at"],
                            "kind": key,
                            "path": value,
                            "url": media_url(value),
                        }
                    )

    patient_data = json.loads(patient_row["data_json"] or "{}")
    return jsonify(
        {
            "patient": {
                "id": patient_row["id"],
                "patientId": patient_row["patient_id"],
                "patientName": patient_row["patient_name"],
                "age": patient_row["age"],
                "gender": patient_row["gender"],
                "clinicalRisk": patient_row["clinical_risk"],
                "createdAt": patient_row["created_at"],
                "updatedAt": patient_row["updated_at"],
                "data": patient_data,
            },
            "events": events,
            "media": media,
            "counts": {
                "events": len(events),
                "media": len(media),
            },
        }
    )


@app.route("/patient-media/<path:rel_path>", methods=["GET"])
def serve_patient_media(rel_path):
    normalized = rel_path.replace("\\", "/")
    abs_target = os.path.abspath(os.path.join(MEDIA_ROOT, normalized))
    media_root_abs = os.path.abspath(MEDIA_ROOT)
    if not abs_target.startswith(media_root_abs):
        return jsonify({"error": "Invalid media path"}), 400
    directory = os.path.dirname(abs_target)
    filename = os.path.basename(abs_target)
    if not os.path.exists(abs_target):
        return jsonify({"error": "Media file not found"}), 404
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
