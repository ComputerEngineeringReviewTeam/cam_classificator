import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tsf

from model import FragmentClassifier


DATA_PATH = '../data/new'           # Path to the folder containing the full images in 2 subfolders - 1 per class
TARGET_SIZE = (224, 224)            # Size to which the images will be resized
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = '../model/model.pth'   # Path to save the trained model
TRAIN = True                        # Set to True to train the model
SAVE_MODEL = True                   # Set to True to save the trained model
TEST = False                        # Set to True to test the model
LOAD_MODEL = False                  # Set to True to load the trained model


transforms = tsf.Compose([tsf.Resize(TARGET_SIZE), tsf.ToTensor()]) # Resize and convert the images to tensors
dataset = ImageFolder(root=DATA_PATH, transform=transforms)         # Load the images, 1 subfolder per class
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = FragmentClassifier().to(device=DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


if TRAIN:
    model.train()
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device=DEVICE), y.to(device=DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
    if SAVE_MODEL:
        torch.save(model.state_dict(), MODEL_PATH)


if TEST:
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device=DEVICE), y.to(device=DEVICE)
            y_pred = model(x)
            print(f"Prediction: {y_pred.argmax(dim=1)} Ground Truth: {y}")
