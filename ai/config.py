import os
import torch

from enum import Enum
from typing import Literal

import ai.utils.filters as flt


# Environment config

class Modes(Enum):
    CLASSIFIER = 1
    REGRESSOR = 2
    BOTH = 3


# Paths
os.environ["CAM_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM_ROOT = os.environ["CAM_ROOT"]
DATA_DIR = os.path.join(CAM_ROOT, "data")
LABELS_PATH = os.path.join(DATA_DIR, "data.json")
IMG_DIR = os.path.join(DATA_DIR, "photos")
MODEL_PATH = os.path.join(CAM_ROOT, "ai", "saved_models", "camnet.pth")
MODELS_DIR = os.path.join(CAM_ROOT, "ai", "saved_models_conv")

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cuda available?", DEVICE)

# Model config
MODEL_NAME = "resnet18"  # Model name from torchvision.models
DROPOUT = 0.2  # Dropout rate for torch.nn.Dropout layers
FEATURES = 256  # Number of features in the deep layer
NUM_AUX_INPUTS = 0
MODE: Literal['both', 'regressor', 'classifier'] = Modes.REGRESSOR  # Controls model, loss and metrics

# Data config
TARGET_SIZE = (224, 224)  # (width, height) to resize images to
ONLY_GOOD = True
BALANCED_SPLIT = False

# Config for the image filters (applied before feeding the image into the network)
datasetFilterSet = flt.Filters()
(datasetFilterSet
 # .addFilter(flt.StandardPILFilter(flt.SHARPEN))

 .addFilter(flt.StandardPILFilter(flt.CONTOUR))
 # .addFilter(flt.StandardPILFilter(flt.EDGE_ENHANCE))
 # .addFilter(flt.MatrixFilter([0, -1, 0, -1, 4, -1, 0, -1, 0]))
 # .addFilter(flt.MatrixFilter([-1, -1, -1, -1, 8, -1, -1, -1, -1]))
 )
DISPLAY_IMAGES_BEFORE_FILTERS = False
DISPLAY_IMAGES_AFTER_FILTERS = False

# Training config
TRAIN = True  # Set to True to train the model
SAVE_MODEL = True  # Set to True to save the model to MODEL_PATH
LEARNING_RATE = 1e-3  # Learning rate for optimizer
EPOCHS = 10  # Number of epochs to train for
BATCH_SIZE = 32  # Dataloader batch size
TRAIN_FRACTION = 0.8  # Fraction of data to use for training
RANDOM_SEED = 12345  # Random seed used to split data into train/test sets, used for reproducibility

# Test config
TEST = True  # Set to True to test the model
LOAD_MODEL = False  # Set to True to load a model from MODEL_PATH
BINARY_ACCURACY_THRESHOLD = 0.5  # Threshold for binary accuracy metric
