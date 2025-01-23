import torch

LEARNING_RATE = 1e-3
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
TRAIN = False
MODEL_NAME = "resnet50"
