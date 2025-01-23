import torch

# Data config
ROOT_DIR = "../../"
TARGET_SIZE = (224, 224)
BATCH_SIZE = 8

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training config
TRAIN = False
LEARNING_RATE = 1e-3
EPOCHS = 5
MODEL_NAME = "resnet50"

# Test config
TEST = False
METRIC_TASK = 'multilabel'
NUM_LABELS = 5

