import os
import time
import torch

CAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(CAM_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "new", "train")   # Path to the folder with full train images in 2 subfolders - 1 per class
TEST_DATA_PATH = os.path.join(DATA_ROOT, "new", "test")     # Path to the folder with full test images in 2 subfolders - 1 per class
CAM_CLASSIFIER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH = os.path.join(CAM_CLASSIFIER_ROOT, "models", "fragments", "model99.pth")     # Path to the trained model
CLASS_1_OUTPUT_PATH = os.path.join(CAM_CLASSIFIER_ROOT, "data", "class_1") # Path to save the images classified as class 1
CLASS_2_OUTPUT_PATH = os.path.join(CAM_CLASSIFIER_ROOT, "data", "class_2") # Path to save the images classified as class 2
TARGET_SIZE = (224, 224)                # Size to which the images will be resized
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(CAM_ROOT, "models", "fragments", "model" + str(time.time()) + ".pth")
TRAIN = True                            # Set to True to train the model
SAVE_MODEL = True                       # Set to True to save the trained model
TEST = True                             # Set to True to test the model
LOAD_MODEL = True                       # Set to True to load the trained model
