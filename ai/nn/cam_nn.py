import torchmetrics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from ai.dataset.cam_label import JsonLabelLoader
from ai.nn.config import *
from ai.nn.camnet import CamNet
from ai.nn.custom_loss import CustomLoss
from ai.dataset.dataset_helpers import train_test_datasets


def train(model: torch.nn.Module, device: str, loss_fn: torch.nn.Module, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, epochs: int):
    print('Training...')
    for epoch in range(epochs):

        for i, ((image, scale), label) in enumerate(dataloader):
            image, scale, label = image.to(device), scale.to(device), label.to(device)
            binary_output, regression_output = model((image, scale))  # model.random_trans(image)
            binary_target = label[:, -1].unsqueeze(1)
            regression_target = label[:, 1:]
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} loss: {loss.item()}")
    print('Finished Training')


def test(model: torch.nn.Module, device: str, dataloader: DataLoader, metric: torchmetrics.Metric):
    with torch.no_grad():
        for i, ((image, scale), label) in enumerate(dataloader):
            image, scale, label = image.to(device), scale.to(device), label.to(device)
            binary_output, regression_output = model((image, scale))
            binary_target = label[:, -1].unsqueeze(1)
            regression_target = label[:, 1:]

            metric(binary_output, binary_target)


if __name__ == '__main__':
    tsfms = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
    ])

    label_loader = JsonLabelLoader()  # Loads data from JSON file
    train_dataset, test_dataset = train_test_datasets(all_labels=label_loader.load(LABELS_PATH),  # Create train / test datasets
                                                      img_dir=IMG_DIR,                            # with data split according to
                                                      train_fraction=TRAIN_FRACTION,              # TRAIN_FRACTION
                                                      transform=tsfms)                            # TRAIN_FRACTION=0.8 => 80% of data is training data

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)  # Configured DataLoader for loading
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # train and test data in batches

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
        if SAVE_MODEL:
            torch.save(model.state_dict(), MODEL_PATH)

    if TEST:
        if LOAD_MODEL:
            model = CamNet(model_name=MODEL_NAME, pretrained=True, num_aux_inputs=1).to(device=DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        test(model, DEVICE, test_loader, metric)
        print(f"Accuracy: {metric.compute()}")
