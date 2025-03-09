import torch

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data config
TARGET_SIZE = (224, 224)# (width, height) to resize images to

# Model config
MODEL_NAME = "resnet50" # Model name from torchvision.models
DROPOUT = 0.2           # Dropout rate for torch.nn.Dropout layers
FEATURES = 256          # Number of features in the deep layer

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
BINARY_ACCURACY_THRESHOLD = 0.5 # Threshold for binary accuracy metric

# Snapshot config
SNAPSHOT = True
SNAPSHOT_PATH = "snapshot.txt" # Path to save snapshot to