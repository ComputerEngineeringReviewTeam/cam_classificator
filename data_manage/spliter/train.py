import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as tsf
from torchmetrics import Accuracy

import config as conf
from model import FragmentClassifier

train_dataset = ImageFolder(root=conf.TRAIN_DATA_PATH, transform=tsf.Compose([tsf.Resize(conf.TARGET_SIZE),
                                                                              tsf.RandomHorizontalFlip(),
                                                                              tsf.RandomVerticalFlip(),
                                                                              tsf.RandomRotation(degrees=15),
                                                                              tsf.ToTensor()]))
test_dataset = ImageFolder(root=conf.TEST_DATA_PATH,
                           transform=tsf.Compose([tsf.Resize(conf.TARGET_SIZE), tsf.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = FragmentClassifier().to(device=conf.DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR)

train_acc = Accuracy(task='multiclass', num_classes=2).to(device=conf.DEVICE)
test_acc = Accuracy(task='multiclass', num_classes=2).to(device=conf.DEVICE)

if conf.TRAIN:
    model.train()
    for epoch in range(conf.EPOCHS):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device=conf.DEVICE), y.to(device=conf.DEVICE)
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
    if conf.SAVE_MODEL:
        torch.save(model.state_dict(), conf.MODEL_PATH)

results = [0, 0]
if conf.TEST:
    if conf.LOAD_MODEL:
        model.load_state_dict(torch.load(conf.TRAINED_MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device=conf.DEVICE), y.to(device=conf.DEVICE)
            y_pred = model(x)
            test_acc(y_pred, y)
            results[y_pred.argmax()] += 1
            if i % 10 == 0:
                print(f"Prediction: {y_pred.argmax(dim=1)} Ground Truth: {y}")
    print(f'Test Accuracy: {test_acc.compute()}')
    print(f'Good: {results[1]}, Bad: {results[0]}')
