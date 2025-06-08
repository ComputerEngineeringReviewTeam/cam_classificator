import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ai.config import EPOCHS, DEVICE, LABELS_PATH, IMG_DIR, TRAIN_FRACTION, RANDOM_SEED, BALANCED_SPLIT, \
    datasetFilterSet, BATCH_SIZE
from ai.dataset.cam_dataset import CamDataset
from ai.dataset.cam_label import JsonLabelLoader
from ai.metrics.metric_collector import CamMetricCollector
from ai.metrics.metrics_colls import MetricsCollections
from ai.utils.dataset_helpers import prepare_tensors, sample_labels, to_dataloaders
from ai.utils.transforms import CamTransforms


class ConvolutionalRegressionNet(nn.Module):
    def __init__(self):
        super(ConvolutionalRegressionNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # -> [32, 112, 112]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # -> [64, 56, 56]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # -> [128, 28, 28]

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),  # -> [256, 14, 14]
        )

        self.flattened_size = 256 * 14 * 14

        self.regressor = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        out = self.regressor(x)
        return out


def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          epochs: int = EPOCHS,
          device: str = DEVICE):
    print('Training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            regression_output = model(image)
            loss = loss_fn(regression_output, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
    print('Finished Training')


def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         metric_aggregator: CamMetricCollector,
         device: str = DEVICE):
    with torch.no_grad():
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            regression_output = model(image)

            metric_aggregator(None, regression_output, None, regression_target)


# Przykładowe użycie:
if __name__ == "__main__":
    train_tsfms = CamTransforms.Train.grayscale
    test_tsfms = CamTransforms.Test.grayscale
    data_df = JsonLabelLoader().load(LABELS_PATH)

    train_data, test_data = sample_labels(data_df, TRAIN_FRACTION, RANDOM_SEED, balanced=BALANCED_SPLIT)
    train_dataset = CamDataset(train_data, IMG_DIR, train_tsfms, datasetFilterSet)
    test_dataset = CamDataset(test_data, IMG_DIR, test_tsfms, datasetFilterSet)
    train_loader, test_loader = to_dataloaders(train_dataset, test_dataset, batch_size=BATCH_SIZE)

    model = ConvolutionalRegressionNet().to(DEVICE)
    train(model, train_loader)
    metric_aggregator = MetricsCollections.only_regressor

    test(model, test_loader, metric_aggregator, DEVICE)
    metric_aggregator.print_metrics()
