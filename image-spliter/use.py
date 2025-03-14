import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tsf
import os
import shutil

from model import FragmentClassifier

DATA_PATH = '../data/new'               # Path to the folder containing the full images in 2 subfolders - 1 per class
MODEL_PATH = '../models/model.pth'      # Path to the trained model
TARGET_SIZE = (224, 224)                # Size to which the images will be resized
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_1_OUTPUT_PATH = '../data/class_1' # Path to save the images classified as class 1
CLASS_2_OUTPUT_PATH = '../data/class_2' # Path to save the images classified as class 2


transforms = tsf.Compose([tsf.Resize(TARGET_SIZE), tsf.ToTensor()]) # Resize and convert the images to tensors
dataset = ImageFolder(root=DATA_PATH, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
model = FragmentClassifier().load_state_dict(torch.load(MODEL_PATH)).to(device=DEVICE)

# Create the output directories if they don't exist
os.makedirs(CLASS_1_OUTPUT_PATH, exist_ok=True)
os.makedirs(CLASS_2_OUTPUT_PATH, exist_ok=True)

def save_image_to_output(image_path: str, output_path: str):
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(output_path, image_name))
    print(f"Image: {image_name} Saved to: {output_path}")

# Run the model on the images and save them to the appropriate output directory
model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device=DEVICE), y.to(device=DEVICE)
        y_pred = model(x)
        print(f"Prediction: {y_pred.argmax(dim=1)} Ground Truth: {y}")

        if y_pred.argmax(dim=1) == 0:
            output_path = CLASS_1_OUTPUT_PATH
        else:
            output_path = CLASS_2_OUTPUT_PATH
        save_image_to_output(dataset.samples[i][0], output_path)
