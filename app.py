# app.py
# Flask application — thin routing layer only.
# Business logic lives in: models.py, analysis.py, session.py

import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

import analysis
import models
import session as sess

# ── App setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMPL_DIR = os.path.join(ROOT_DIR, "templates")

app = Flask(__name__, template_folder=TMPL_DIR)

# Allow the frontend to call /api/* from ANY origin.
# This is needed when the HTML is served by VS Code Live Server, PHP server,
# or opened as a plain file — rather than by Flask itself.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Load ML models at startup (silent no-op if torch is not installed)
models.load_models()


# ── Helper: decode a base64 JPEG/PNG frame sent from the browser ──────────────
def _decode_frame(b64_string):
    """Decode a data-URL or raw base64 string into a BGR numpy array."""
    raw   = b64_string.split(",")[-1]   # strip "data:image/jpeg;base64," prefix
    data  = base64.b64decode(raw)
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cv2.imdecode returned None — invalid image data")
    return frame


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "active":        sess.is_active(),
        "frames":        len(sess.get_logs()),
        "models_loaded": models.REAL_MODELS_LOADED,
    })


@app.route("/api/start", methods=["POST"])
def api_start():
    try:
        session_id = sess.start()
        analysis.reset_simulation()
        return jsonify({"status": "started", "session_id": session_id})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if not sess.is_active():
        return jsonify({"error": "No active session"}), 400

    body = request.get_json(silent=True)
    if not body or "frame" not in body:
        return jsonify({"error": "Missing 'frame' key in JSON body"}), 400

    try:
        frame = _decode_frame(body["frame"])
    except Exception as e:
        return jsonify({"error": f"Frame decode failed: {e}"}), 400

    try:
        if models.REAL_MODELS_LOADED:
            face_emo, face_conf, voice_emo, voice_conf, gaze = \
                analysis.analyze_frame_real(frame, models.MODELS)
        else:
            face_emo, face_conf, voice_emo, voice_conf, gaze = \
                analysis.analyze_frame_simulation(frame)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {e}"}), 500

    entry = sess.log_frame(face_emo, face_conf, voice_emo, voice_conf, gaze)

    return jsonify({
        "time_sec":     entry["time_sec"],
        "face_emo":     face_emo,
        "face_conf":    round(face_conf, 3),
        "voice_emo":    voice_emo,
        "voice_conf":   round(voice_conf, 3),
        "gaze":         gaze,
        "frames_total": len(sess.get_logs()),
    })


@app.route("/api/end", methods=["POST"])
def api_end():
    if not sess.is_active():
        return jsonify({"error": "No active session"}), 400

    result = sess.end()
    return jsonify(result)


@app.route("/api/download")
def api_download():
    csv_path = sess.latest_csv_path()
    if not csv_path:
        return jsonify({"error": "No log file available yet — run a session first"}), 404

    abs_path = os.path.abspath(csv_path)
    return send_file(
        abs_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=os.path.basename(abs_path),
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(sess.LOGS_DIR, exist_ok=True)
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    mode  = "REAL MODELS" if models.REAL_MODELS_LOADED else "SIMULATION"
    print(f"[TruthLens] http://localhost:{port}  |  mode={mode}  |  debug={debug}")
    app.run(host="0.0.0.0", port=port, debug=debug)