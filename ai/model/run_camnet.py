import torch
import torchmetrics as metrics
import torchvision.transforms as transforms

import ai.model.config as conf
from ai.dataset.cam_label import JsonLabelLoader
from ai.model.camnet import CamNet
from ai.model.camnet2 import CamNet2
from ai.model.custom_loss import CustomLoss
from ai.model.custom_loss2 import CustomLoss2
from ai.dataset.dataset_helpers import describe_dataset, train_test_datasets, to_dataloaders
from ai.model.metrics import BinaryMetrics, CamMetricAggregator, RelativeError


def prepare_tensors(image: torch.Tensor,
                    scale: torch.Tensor,
                    binary_target: torch.Tensor,
                    regression_target: torch.Tensor,
                    device: str):
    """
    Moves tensors to device and reshapes binary_target to have 2 dimensions

    ! Reshaping must be done after moving to device !

    Args:
        image: torch.Tensor
        scale: torch.Tensor
        binary_target: torch.Tensor
        regression_target: torch.Tensor
        device: str

    Returns:
        image: torch.Tensor
        scale: torch.Tensor
        binary_target: torch.Tensor
        regression_target: torch.Tensor
    """

    image, scale = image.to(device), scale.to(device)
    binary_target, regression_target = binary_target.to(device), regression_target.to(device)
    binary_target = binary_target.unsqueeze(1)
    return image, scale, binary_target, regression_target


def train(model: torch.nn.Module,
          loss_fn: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int = conf.EPOCHS,
          device: str = conf.DEVICE):
    print('Training...')
    for epoch in range(epochs):
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            binary_output, regression_output = model((image, scale))  # model.random_trans(image)
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
    print('Finished Training')


def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         metric_aggregator: CamMetricAggregator,
         device: str = conf.DEVICE):
    with torch.no_grad():
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            binary_output, regression_output = model((image, scale))

            metric_aggregator(binary_output, regression_output, binary_target, regression_target)


def run_camnet():
    bin_thr = conf.BINARY_ACCURACY_THRESHOLD

    # Transforms for the images used
    tsfms = transforms.Compose([
        transforms.Resize(conf.TARGET_SIZE),
    ])

    # Create datasets and dataloaders for loading the data using shortcut functions
    train_dataset, test_dataset = train_test_datasets(JsonLabelLoader().load(conf.LABELS_PATH), # Loads from a JSON file
                                                      img_dir=conf.IMG_DIR,
                                                      train_fraction=conf.TRAIN_FRACTION,
                                                      transform=tsfms,
                                                      random_state=conf.RANDOM_SEED,
                                                      balanced=True)
    train_loader, test_loader = to_dataloaders(train_dataset, test_dataset, batch_size=conf.BATCH_SIZE)

    # Create the model, optimizer and loss function
    model = CamNet2(model_name=conf.MODEL_NAME,
                    pretrained=True,
                    num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    loss_fn = CustomLoss2()

    # Create the metrics used for testing the model
    metric_aggregator = CamMetricAggregator(
        classification_metrics=[
            metrics.classification.BinaryAccuracy(threshold=bin_thr).to(device=conf.DEVICE),
            metrics.classification.BinaryRecall(threshold=bin_thr).to(device=conf.DEVICE),
            metrics.classification.BinaryPrecision(threshold=bin_thr).to(device=conf.DEVICE),
            BinaryMetrics()
        ],  regression_metrics=[
            metrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            RelativeError()
        ])

    # Run training loop
    if conf.TRAIN:
        describe_dataset(train_dataset)
        model.train()
        train(model, loss_fn, train_loader, optimizer)
        if conf.SAVE_MODEL:
            torch.save(model.state_dict(), conf.MODEL_PATH)

    # Run testing loop
    if conf.TEST:
        describe_dataset(test_dataset)
        if conf.LOAD_MODEL:
            model = CamNet2(model_name=conf.MODEL_NAME,
                            pretrained=True,
                            num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
            model.load_state_dict(torch.load(conf.MODEL_PATH, weights_only=True))
        model.eval()
        test(model, test_loader, metric_aggregator)
        metric_aggregator.print_metrics()

if __name__ == '__main__':
    run_camnet()
