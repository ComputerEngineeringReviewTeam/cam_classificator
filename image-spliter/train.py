import os
import time
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tsf
from torchmetrics import Accuracy

from model import FragmentClassifier

CAM_CLASSIFIER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(CAM_CLASSIFIER_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "new", "train")   # Path to the folder with full train images in 2 subfolders - 1 per class
TEST_DATA_PATH = os.path.join(DATA_ROOT, "new", "test")     # Path to the folder with full test images in 2 subfolders - 1 per class
TARGET_SIZE = (224, 224)                # Size to which the images will be resized
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(CAM_CLASSIFIER_ROOT, "models", "fragments", "model" + str(time.time()) + ".pth")
TRAIN = True                            # Set to True to train the model
SAVE_MODEL = True                       # Set to True to save the trained model
TEST = True                             # Set to True to test the model
LOAD_MODEL = True                       # Set to True to load the trained model


train_dataset = ImageFolder(root=TRAIN_DATA_PATH, transform=tsf.Compose([tsf.Resize(TARGET_SIZE),
                                                                         tsf.RandomHorizontalFlip(),
                                                                         tsf.RandomVerticalFlip(),
                                                                         tsf.RandomRotation(degrees=15),
                                                                         tsf.ToTensor()]))
test_dataset = ImageFolder(root=TEST_DATA_PATH, transform=tsf.Compose([tsf.Resize(TARGET_SIZE), tsf.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = FragmentClassifier().to(device=DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_acc = Accuracy(task='multiclass', num_classes=2).to(device=DEVICE)
test_acc = Accuracy(task='multiclass', num_classes=2).to(device=DEVICE)


if TRAIN:
    model.train()
    for epoch in range(EPOCHS):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device=DEVICE), y.to(device=DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_acc(y_pred, y)
            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}')
        print(f'Epoch {epoch}, Train Accuracy: {train_acc.compute()}')
        train_acc.reset()
    if SAVE_MODEL:
        torch.save(model.state_dict(), MODEL_PATH)


results = [0, 0]
if TEST:
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device=DEVICE), y.to(device=DEVICE)
            y_pred = model(x)
            test_acc(y_pred, y)
            results[y_pred.argmax()] += 1
            if i % 10 == 0:
                print(f"Prediction: {y_pred.argmax(dim=1)} Ground Truth: {y}")
    print(f'Test Accuracy: {test_acc.compute()}')
    print(f'Good: {results[1]}, Bad: {results[0]}')
