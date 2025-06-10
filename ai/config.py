import torch

import ai.utils.filters as flt
from ai.utils.modes import Modes


# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Cuda available?", DEVICE)

# Model config
MODEL_NAME = "resnet18"  # Model name from torchvision.models
DROPOUT = 0.2  # Dropout rate for torch.nn.Dropout layers
FEATURES = 256  # Number of features in the deep layer
NUM_AUX_INPUTS = 0
MODE: Modes = Modes.REGRESSOR  # Controls model, loss and metrics

# Data config
TARGET_SIZE = (224, 224)  # (width, height) to resize images to
ONLY_GOOD = True
BALANCED_SPLIT = False

# Config for the image filters (applied before feeding the image into the network)
datasetFilterSet = flt.Filters()
(datasetFilterSet
 # .addFilter(flt.StandardPILFilter(flt.SHARPEN))
 # .addFilter(flt.StandardPILFilter(flt.CONTOUR))
 # .addFilter(flt.StandardPILFilter(flt.EDGE_ENHANCE))
 # .addFilter(flt.MatrixFilter([0, -1, 0, -1, 4, -1, 0, -1, 0]))
 # .addFilter(flt.MatrixFilter([-1, -1, -1, -1, 8, -1, -1, -1, -1]))
 )
DISPLAY_IMAGES_BEFORE_FILTERS = False
DISPLAY_IMAGES_AFTER_FILTERS = False

# Training config
LEARNING_RATE = 1e-3  # Learning rate for optimizer
EPOCHS = 10  # Number of epochs to train for
BATCH_SIZE = 32  # Dataloader batch size
TRAIN_FRACTION = 0.8  # Fraction of data to use for training
RANDOM_SEED = 12345  # Random seed used to split data into train/test sets, used for reproducibility

# Test config
BINARY_ACCURACY_THRESHOLD = 0.5  # Threshold for binary accuracy metric

# ai.runners.run_camnet.py config
TRAIN = True  # Set to True to train the model
SAVE_MODEL = True  # Set to True to save the model to SAVED_MODEL_PATH
TEST = True  # Set to True to test the model
LOAD_MODEL = False  # Set to True to load a model from SAVED_MODEL_PATH