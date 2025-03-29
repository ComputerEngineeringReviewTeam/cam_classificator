import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tsf
import os
import shutil

import data_manage.spliter.config as conf
from model import FragmentClassifier

transforms = tsf.Compose([tsf.Resize(conf.TARGET_SIZE), tsf.ToTensor()])  # Resize and convert the images to tensors
dataset = ImageFolder(root=conf.DATA_ROOT, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
model = FragmentClassifier().to(device=conf.DEVICE)
model.load_state_dict(torch.load(conf.TRAINED_MODEL_PATH, weights_only=True))

# Create the output directories if they don't exist
os.makedirs(conf.CLASS_1_OUTPUT_PATH, exist_ok=True)
os.makedirs(conf.CLASS_2_OUTPUT_PATH, exist_ok=True)


def save_image_to_output(image_path: str, output_path: str):
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(output_path, image_name))
    # print(f"Image: {image_name} Saved to: {output_path}")


# Run the model on the images and save them to the appropriate output directory
model.eval()
good, bad = 0, 0
with torch.no_grad():
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device=conf.DEVICE), y.to(device=conf.DEVICE)
        y_pred = model(x)

        if y_pred.argmax(dim=1) == 0:
            output_path = conf.CLASS_1_OUTPUT_PATH
            bad += 1
        else:
            output_path = conf.CLASS_2_OUTPUT_PATH
            good += 1
        save_image_to_output(dataset.samples[i][0], output_path)
        if i % 10 == 0:
            print(f"Processed {i} images")
            print(f"Good: {good}, Bad: {bad}")
