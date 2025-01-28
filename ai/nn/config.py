import torch
import os

# Data config
CAM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CAM_ROOT"] = CAM_ROOT
os.environ["DATA_DIR"] = os.path.join(os.environ["CAM_ROOT"], "data")
DATA_DIR = os.environ["DATA_DIR"]
LABELS_PATH = os.path.join(DATA_DIR, "data.json")
IMG_DIR = os.path.join(DATA_DIR, "photos")
MODEL_PATH = os.path.join(CAM_ROOT, "model.pth")
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8
TRAIN_FRACTION = 0.8

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cuda available?", torch.cuda.is_available())

# Training config
TRAIN = True
SAVE_MODEL = True
LEARNING_RATE = 1e-3
EPOCHS = 5
MODEL_NAME = "resnet50"
DROPOUT = 0.2
FEATURES = 256

# Test config
TEST = True
LOAD_MODEL = False
BINARY_ACCURACY_THRESHOLD = 0.5
