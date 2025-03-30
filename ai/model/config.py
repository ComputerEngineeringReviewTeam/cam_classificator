import os
import torch
import utils.filters as flt

datasetFilterSet = flt.Filters()

# Paths
os.environ["CAM_ROOT"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM_ROOT = os.environ["CAM_ROOT"]
DATA_DIR = os.path.join(CAM_ROOT, "data")
LABELS_PATH = os.path.join(DATA_DIR, "data.json")
IMG_DIR = os.path.join(DATA_DIR, "photos")
MODEL_PATH = os.path.join(CAM_ROOT, "ai", "saved_models", "camnet.pth")
# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cuda available?", torch.cuda.is_available())

# Data config
TARGET_SIZE = (224, 224)  # (width, height) to resize images to

# Model config
MODEL_NAME = "resnet50"  # Model name from torchvision.models
DROPOUT = 0.2            # Dropout rate for torch.nn.Dropout layers
FEATURES = 256           # Number of features in the deep layer

# Training config
TRAIN = True            # Set to True to train the model
SAVE_MODEL = True       # Set to True to save the model to MODEL_PATH
LEARNING_RATE = 1e-3    # Learning rate for optimizer
EPOCHS = 5              # Number of epochs to train for
BATCH_SIZE = 8          # Dataloader batch size
TRAIN_FRACTION = 0.8    # Fraction of data to use for training
RANDOM_SEED = 1234      # Random seed used to split data into train/test sets, used for reproducibility

# Test config
TEST = True             # Set to True to test the model
LOAD_MODEL = False      # Set to True to load a model from MODEL_PATH
BINARY_ACCURACY_THRESHOLD = 0.5  # Threshold for binary accuracy metric

#Config for the image filters (applied before feeding the image into the network)
(datasetFilterSet
 .addFilter(flt.MatrixFilter([0, 0, 0, 0, 1, 0, 0, 0, 0]))
 #.addFilter(flt.MatrixFilter([0, 0, 0, 0, 1, 0, 0, 0, 0]))
 )
