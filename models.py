# models.py
# Attempts to load the trained .pt model files from the project root.
# If torch / ultralytics are not installed, or the files are absent,
# REAL_MODELS_LOADED stays False and the app runs in simulation mode.

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "face_model":       None,
    "face_class_names": None,
    "audio_model":      None,
    "audio_idx_to_class": None,
    "face_detector":    None,
    "device":           None,
}

REAL_MODELS_LOADED = False


def _build_face_cnn(num_classes, torch, nn, F):
    import torch.nn as nn_mod
    import torch.nn.functional as F_mod

    class EmotionCNN(nn_mod.Module):
        def __init__(self, n):
            super().__init__()
            self.conv1   = nn_mod.Conv2d(1, 32, 3, padding=1)
            self.conv2   = nn_mod.Conv2d(32, 64, 3, padding=1)
            self.conv3   = nn_mod.Conv2d(64, 128, 3, padding=1)
            self.pool    = nn_mod.MaxPool2d(2, 2)
            self.dropout = nn_mod.Dropout(0.5)
            self.fc1     = nn_mod.Linear(128 * 6 * 6, 256)
            self.fc2     = nn_mod.Linear(256, n)

        def forward(self, x):
            x = self.pool(F_mod.relu(self.conv1(x)))
            x = self.pool(F_mod.relu(self.conv2(x)))
            x = self.pool(F_mod.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.dropout(F_mod.relu(self.fc1(x)))
            return self.fc2(x)

    return EmotionCNN(num_classes)


def _build_audio_mlp(input_dim, num_classes, nn_mod):
    class AudioMLP(nn_mod.Module):
        def __init__(self):
            super().__init__()
            self.fc1     = nn_mod.Linear(input_dim, 128)
            self.fc2     = nn_mod.Linear(128, 64)
            self.fc3     = nn_mod.Linear(64, num_classes)
            self.dropout = nn_mod.Dropout(0.4)

        def forward(self, x):
            import torch
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    return AudioMLP()


def load_models():
    """Try to load all .pt model files. Safe to call even without torch installed."""
    global REAL_MODELS_LOADED, MODELS

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[models] torch/ultralytics not installed — simulation mode ({e})")
        return

    device = torch.device("cpu")
    MODELS["device"] = device
    loaded_any = False

    face_pt   = os.path.join(ROOT_DIR, "emotion_model.pt")
    audio_pt  = os.path.join(ROOT_DIR, "audio_model_tess.pt")
    detect_pt = os.path.join(ROOT_DIR, "best.pt")

    if os.path.exists(face_pt):
        try:
            ckpt = torch.load(face_pt, map_location=device)
            MODELS["face_class_names"] = ckpt["class_names"]
            model = _build_face_cnn(len(MODELS["face_class_names"]), torch, nn, F)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            MODELS["face_model"] = model
            loaded_any = True
            print(f"[models] face emotion model loaded — classes: {MODELS['face_class_names']}")
        except Exception as e:
            print(f"[models] failed to load emotion_model.pt: {e}")

    if os.path.exists(audio_pt):
        try:
            ckpt = torch.load(audio_pt, map_location=device)
            MODELS["audio_idx_to_class"] = ckpt["idx_to_class"]
            model = _build_audio_mlp(40, len(ckpt["class_to_idx"]), nn)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device).eval()
            MODELS["audio_model"] = model
            loaded_any = True
            print(f"[models] audio emotion model loaded — classes: {MODELS['audio_idx_to_class']}")
        except Exception as e:
            print(f"[models] failed to load audio_model_tess.pt: {e}")

    if os.path.exists(detect_pt):
        try:
            MODELS["face_detector"] = YOLO(detect_pt)
            loaded_any = True
            print("[models] YOLO face detector loaded (best.pt)")
        except Exception as e:
            print(f"[models] failed to load best.pt: {e}")

    REAL_MODELS_LOADED = loaded_any
    if not loaded_any:
        print("[models] no .pt files found — simulation mode")
