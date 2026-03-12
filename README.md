# TruthLens

A real-time emotional analysis web app that uses your webcam to detect facial emotions, track gaze direction, and classify voice emotions — then generates a session report with percentage breakdowns.

## Tech Stack

- **Backend:** Python, Flask, OpenCV
- **Frontend:** React, HTML/CSS
- **ML Models:** PyTorch (CNN for face emotion, MLP for voice), YOLOv8

## Features

- Live webcam feed with real-time emotion overlays
- Face emotion detection (7 classes: neutral, happy, sad, angry, fear, disgust, surprise)
- Gaze direction tracking (left, center, right)
- Voice emotion classification (MFCC-based)
- Session report with percentage breakdowns and final stress state
- CSV log export for every session

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/your-username/truthlens.git
cd truthlens
```

**2. Install dependencies**
```bash
pip install flask flask-cors opencv-python numpy
```

**3. Run**
```bash
python app.py
```

Open `http://localhost:5000` in your browser and allow camera access.

## Project Structure

```
truthlens/
├── app.py              # Flask routes
├── models.py           # ML model loading
├── analysis.py         # Frame analysis logic
├── session.py          # Session state & CSV logging
├── index.html          # React frontend
└── requirements.txt    # Python dependencies
```

## Live Demo

https://truthlens-cjpu.onrender.com

## Notes

- Works in simulation mode out of the box (no model files needed)
- Place `best.pt`, `emotion_model.pt`, and `audio_model_tess.pt` in the root to enable real inference
- Model files are excluded from the repo via `.gitignore` due to size
