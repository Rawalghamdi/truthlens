# analysis.py
# Two analysis paths:
#   analyze_frame_real()       — uses loaded .pt models
#   analyze_frame_simulation() — uses OpenCV Haar cascades + temporal simulation

import random
import cv2
import numpy as np

# ── Haar cascades (always available via opencv-python) ────────────────────────
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_EMOTIONS  = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
VOICE_EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]

# Shared mutable simulation state — reset at session start
_sim = {
    "face_emo":   "neutral",
    "voice_emo":  "neutral",
    "face_conf":  0.72,
    "voice_conf": 0.65,
}


def reset_simulation():
    _sim.update(face_emo="neutral", voice_emo="neutral",
                face_conf=0.72, voice_conf=0.65)


def _drift(value, lo=0.35, hi=0.99, step=0.05):
    return max(lo, min(hi, value + random.uniform(-step, step)))


def _maybe_switch(current, options, prob):
    return random.choice(options) if random.random() < prob else current


# ── Gaze helper (shared by both paths) ───────────────────────────────────────
def _gaze_from_face_roi(gray_face):
    """Return 'left' | 'center' | 'right' | 'unknown' from a grayscale face ROI."""
    h, w = gray_face.shape[:2]
    upper = gray_face[: int(h * 0.6), :]
    eyes = _eye_cascade.detectMultiScale(upper, 1.1, 6, minSize=(20, 20))
    if len(eyes) == 0:
        return "unknown"
    top2 = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    avg_cx = sum(ex + ew / 2 for ex, ey, ew, eh in top2) / len(top2)
    rel = (avg_cx - w / 2) / (w / 2)
    if rel < -0.07:
        return "left"
    if rel > 0.07:
        return "right"
    return "center"


# ── Simulation path ───────────────────────────────────────────────────────────
def analyze_frame_simulation(frame_bgr):
    """
    Real gaze detection from webcam + temporally-smooth emotion simulation.
    Returns (face_label, face_conf, voice_label, voice_conf, gaze).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5)
    face_detected = len(faces) > 0

    # Real eye-based gaze
    gaze = "unknown"
    if face_detected:
        x, y, w, h = faces[0]
        gaze = _gaze_from_face_roi(gray[y: y + h, x: x + w])

    # Smoothly drift emotions
    change_prob = 0.12 if face_detected else 0.0
    _sim["face_emo"]   = _maybe_switch(_sim["face_emo"],   FACE_EMOTIONS,  change_prob)
    _sim["voice_emo"]  = _maybe_switch(_sim["voice_emo"],  VOICE_EMOTIONS, 0.10)
    _sim["face_conf"]  = _drift(_sim["face_conf"],  lo=0.40)
    _sim["voice_conf"] = _drift(_sim["voice_conf"], lo=0.35)

    face_label = _sim["face_emo"]  if face_detected else "no_face"
    face_conf  = _sim["face_conf"] if face_detected else 0.0

    return face_label, face_conf, _sim["voice_emo"], _sim["voice_conf"], gaze


# ── Real model path ───────────────────────────────────────────────────────────
def analyze_frame_real(frame_bgr, models):
    """
    Uses YOLO face detector + EmotionCNN + AudioMLP (or Haar fallback).
    Returns (face_label, face_conf, voice_label, voice_conf, gaze).
    """
    import torch
    import torch.nn.functional as F

    device          = models["device"]
    face_detector   = models.get("face_detector")
    face_model      = models.get("face_model")
    face_classes    = models.get("face_class_names", [])

    face_label, face_conf, gaze = "no_face", 0.0, "unknown"

    # ── Face detection ────────────────────────────────────────────────────────
    boxes = []
    if face_detector:
        results = face_detector(frame_bgr, verbose=False)
        r = results[0]
        if r.boxes and len(r.boxes) > 0:
            h, w = frame_bgr.shape[:2]
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b.astype(int)
                boxes.append((
                    max(0, x1), max(0, y1),
                    min(w, x2), min(h, y2),
                ))
    else:
        # Haar fallback if YOLO weights not present
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in _face_cascade.detectMultiScale(gray, 1.1, 5):
            boxes.append((x, y, x + w, y + h))

    if boxes:
        x1, y1, x2, y2 = boxes[0]
        face_roi = frame_bgr[y1:y2, x1:x2]

        if face_roi.size > 0:
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gaze = _gaze_from_face_roi(gray_roi)

            if face_model and face_classes:
                resized = cv2.resize(gray_roi, (48, 48)).astype("float32") / 255.0
                tensor  = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs   = torch.softmax(face_model(tensor), dim=1)[0]
                    idx     = torch.argmax(probs).item()
                    face_label = face_classes[idx]
                    face_conf  = float(probs[idx])

    # ── Voice (simulated — real mic capture needs a separate audio thread) ────
    _sim["voice_emo"]  = _maybe_switch(_sim["voice_emo"], VOICE_EMOTIONS, 0.10)
    _sim["voice_conf"] = _drift(_sim["voice_conf"], lo=0.35)

    return face_label, face_conf, _sim["voice_emo"], _sim["voice_conf"], gaze
