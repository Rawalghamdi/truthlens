# session.py
# Manages the in-memory session state, CSV writing, and final-state scoring.

import csv
import os
import time
import datetime
from collections import Counter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# ── Session state (module-level singleton) ────────────────────────────────────
_state = {
    "active":     False,
    "logs":       [],       # list of dicts
    "start_time": None,
    "csv_path":   None,
    "session_id": None,
}


def is_active():
    return _state["active"]


def get_logs():
    return _state["logs"]


def get_csv_path():
    return _state["csv_path"]


def start():
    """Initialise a new session. Returns session_id string or raises if already active."""
    if _state["active"]:
        raise RuntimeError("Session already active")

    os.makedirs(LOGS_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(LOGS_DIR, f"truthlens_session_{ts}.csv")

    _state.update(
        active     = True,
        logs       = [],
        start_time = time.time(),
        csv_path   = csv_path,
        session_id = ts,
    )
    return ts


def log_frame(face_emo, face_conf, voice_emo, voice_conf, gaze):
    """Append one analysed frame to the session log. Returns the log entry dict."""
    t = round(time.time() - _state["start_time"], 2)
    entry = dict(
        time_sec   = t,
        face_emo   = face_emo,
        face_conf  = round(face_conf, 4),
        voice_emo  = voice_emo,
        voice_conf = round(voice_conf, 4),
        gaze       = gaze,
    )
    _state["logs"].append(entry)
    return entry


def end():
    """
    End the session, write CSV, compute percentages + final state.
    Returns a result dict. Safe to call even with zero frames.
    """
    _state["active"] = False
    logs = _state["logs"]

    if not logs:
        return {
            "final_state":   "No Data",
            "face_percent":  {},
            "gaze_percent":  {},
            "voice_percent": {},
            "total_frames":  0,
            "duration_sec":  0,
            "csv_file":      None,
        }

    total    = len(logs)
    duration = round(logs[-1]["time_sec"], 1)

    face_pct  = _to_percent(Counter(r["face_emo"]  for r in logs), total)
    voice_pct = _to_percent(Counter(r["voice_emo"] for r in logs), total)
    gaze_pct  = _to_percent(Counter(r["gaze"]      for r in logs), total)

    final_state = _determine_final_state(logs)

    # Write CSV
    csv_file = None
    try:
        fields = ["time_sec", "face_emo", "face_conf", "voice_emo", "voice_conf", "gaze"]
        with open(_state["csv_path"], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(logs)
        csv_file = os.path.basename(_state["csv_path"])
        print(f"[session] log saved → {_state['csv_path']}")
    except Exception as e:
        print(f"[session] CSV write error: {e}")

    return {
        "final_state":   final_state,
        "face_percent":  face_pct,
        "gaze_percent":  gaze_pct,
        "voice_percent": voice_pct,
        "total_frames":  total,
        "duration_sec":  duration,
        "csv_file":      csv_file,
    }


def latest_csv_path():
    """Return the absolute path to the most recent CSV log, or None."""
    # Prefer the current session's file
    if _state["csv_path"] and os.path.exists(_state["csv_path"]):
        return _state["csv_path"]
    # Fallback: newest file in logs/
    if os.path.isdir(LOGS_DIR):
        files = sorted(
            [f for f in os.listdir(LOGS_DIR) if f.endswith(".csv")],
            reverse=True,
        )
        if files:
            return os.path.join(LOGS_DIR, files[0])
    return None


# ── Scoring helpers ───────────────────────────────────────────────────────────
def _stress_score(row):
    score = 0
    if row["face_emo"]  in {"fear", "angry", "sad", "disgust"} and row["face_conf"]  >= 0.40:
        score += 1
    if row["voice_emo"] in {"fear", "angry", "sad"}            and row["voice_conf"] >= 0.40:
        score += 1
    if row["gaze"] in {"left", "right"}:
        score += 1
    return score


def _determine_final_state(logs):
    scores = [_stress_score(r) for r in logs]
    high_stress_pct = sum(1 for s in scores if s >= 2) / len(scores)
    avg_stress      = sum(scores) / len(scores)

    if high_stress_pct > 0.40 or avg_stress >= 1.5:
        return "High Stress"
    if high_stress_pct > 0.20 or avg_stress >= 0.8:
        return "Mild Stress"

    dominant_face = Counter(r["face_emo"] for r in logs).most_common(1)[0][0]
    if dominant_face == "happy":
        return "Positive / Calm"
    if dominant_face == "neutral":
        return "Neutral / Composed"
    return "Relaxed"


def _to_percent(counter, total):
    if total == 0:
        return {}
    return {k: round(v / total * 100, 1) for k, v in counter.items()}
