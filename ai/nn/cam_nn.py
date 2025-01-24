from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from ai.dataset.cam_label import JsonLabelLoader
from ai.nn.config import *
from ai.nn.camnet import CamNet
from ai.nn.custom_loss import CustomLoss
from ai.dataset.dataset_helpers import train_test_datasets


def train(model, device, loss_fn, dataloader, optimizer, epochs):
    for epoch in range(epochs):
        for i, ((image, scale), label) in enumerate(dataloader):
            image, scale, label = image.to(device), scale.to(device), label.to(device)
            result = model((image, scale))
            loss = loss_fn(result, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, device, dataloader, metric):
    with torch.no_grad():
        for i, ((image, scale), label) in enumerate(dataloader):
            image, scale, label = image.to(device), scale.to(device), label.to(device)
            result = model((image, scale))
            metric(result, label)


if __name__ == '__main__':
    tsfms = Compose([           # No need to use the ToTensor() transform as the image is already a tensor
        Resize(*TARGET_SIZE),
    ])

    label_loader = JsonLabelLoader()
    train_dataset, test_dataset = train_test_datasets(all_labels=label_loader.load(LABELS_PATH),
                                                      img_dir=IMG_DIR,
                                                      train_fraction=TRAIN_FRACTION,
                                                      transform=tsfms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)      # Configured DataLoader for loading
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)        # train and test data in batches

    model = (CamNet(model_name=MODEL_NAME,
                   pretrained=True,
                   num_aux_inputs=1)
             .to(device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CustomLoss()

    metric = Accuracy(task=METRIC_TASK, num_labels=NUM_LABELS)

    if TRAIN:
        model.train()
        train(model, DEVICE, loss_fn, train_loader, optimizer, EPOCHS)

    if TEST:
        model.eval()
        test(model, DEVICE, test_loader, metric)
        print(f"Accuracy: {metric.compute()}")
