import os

CAM_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(CAM_ROOT, "data")
LABELS_PATH = os.path.join(DATA_DIR, "data.json")
IMG_DIR = os.path.join(DATA_DIR, "photos")

SAVED_MODELS_DIR = os.path.join(CAM_ROOT, "ai", "saved_models")
SAVED_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "model.pth")