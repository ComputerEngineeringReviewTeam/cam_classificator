import os

# Environment config
os.environ["CAM_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["DATA_DIR"] = os.path.join(os.environ["CAM_ROOT"], "data")

# Paths
LABELS_PATH = os.path.join(os.environ["CAM_ROOT"], "data", "data.json")
IMG_DIR = os.path.join(os.environ["CAM_ROOT"], "data", "photos")
MODEL_PATH = os.path.join(os.environ["CAM_ROOT"], "model.pth")
LOGS_DIR = os.path.join(os.environ["CAM_ROOT"], "logs")
