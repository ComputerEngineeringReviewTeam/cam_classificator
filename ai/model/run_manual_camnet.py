import os.path
import torch
import torchmetrics as metrics
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy

import ai.model.config as conf
from ai.dataset.cam_dataset import CamDataset
from ai.dataset.cam_label import JsonLabelLoader
from ai.model.camnet import CamNet
from ai.model.camnet2 import CamNet2
from ai.model.convo import CamNetConv
from ai.model.custom_loss import CustomLoss
from ai.model.custom_loss2 import CustomLoss2
from ai.dataset.dataset_helpers import train_test_datasets, to_dataloaders, describe_dataset, sample_labels
from ai.model.metrics import CamMetricAggregator, RelativeError, BinaryMetrics, AverageRelativeError
from run_camnet import train, test


def run_manual_camnet():
    BIN_THR = conf.BINARY_ACCURACY_THRESHOLD
    MODELS_DIR = os.path.join(conf.CAM_ROOT, "ai", "saved_models")

    # Transforms for the images used
    train_tsfms = transforms.Compose([
        transforms.Resize(conf.TARGET_SIZE),
        transforms.RandomRotation(30, fill=255),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),
        transforms.Grayscale(num_output_channels=3)
    ])
    test_tsfms = transforms.Compose([
        transforms.Resize(conf.TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3)
    ])

    # Create datasets and dataloaders for loading the data using shortcut functions
    data_df = JsonLabelLoader().load(conf.LABELS_PATH)
    train_data, test_data = sample_labels(data_df, conf.TRAIN_FRACTION, conf.RANDOM_SEED, balanced=False)
    train_dataset = CamDataset(train_data, conf.IMG_DIR, train_tsfms, conf.datasetFilterSet)
    test_dataset = CamDataset(test_data, conf.IMG_DIR, test_tsfms, conf.datasetFilterSet)
    train_loader, test_loader = to_dataloaders(train_dataset, test_dataset, batch_size=conf.BATCH_SIZE)

    # Create the model, optimizer and loss function
    model = CamNet2(model_name=conf.MODEL_NAME,
                    pretrained=True,
                    num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
    # model = CamNetConv().to(device=conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    loss_fn = CustomLoss2()
    # loss_fn = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.LRScheduler(optimizer=optimizer,)

    # Create the metrics used for testing the model
    metric_aggregator = CamMetricAggregator(
        classification_metrics=[
            # BinaryMetrics(),
            # metrics.classification.BinaryAccuracy(threshold=BIN_THR).to(device=conf.DEVICE),
            # metrics.classification.BinaryRecall(threshold=BIN_THR).to(device=conf.DEVICE),
            # metrics.classification.BinaryPrecision(threshold=BIN_THR).to(device=conf.DEVICE),
        ], regression_metrics=[
            metrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            metrics.regression.MeanAbsoluteError().to(device=conf.DEVICE),  # average error
            metrics.regression.NormalizedRootMeanSquaredError().to(device=conf.DEVICE), # standard deviation
            metrics.regression.R2Score().to(device=conf.DEVICE),    # ~accuracy
            RelativeError(),
            AverageRelativeError()
        ])

    # Run the manual loop
    while True:
        input_str = input("Enter command: ")
        if input_str == "q":
            break
        elif input_str == "s":
            filename = input("Enter filename: ")
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, filename + ".pth"))
        elif input_str == 'l':
            filename = input("Enter filename: ")
            try:
                new_model = CamNet2(model_name=conf.MODEL_NAME,
                                    pretrained=True,
                                    num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
                # new_model = CamNetConv().to(device=conf.DEVICE)
                new_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, filename), weights_only=True))
                model = new_model
            except Exception as e:
                print(e)
        elif input_str == 'tr':
            try:
                epochs = int(input("Enter number of epochs: "))
                model.train()
                train(model, loss_fn, train_loader, optimizer, epochs=epochs)
            except ValueError as e:
                print(e)
        elif input_str == 'te':
            metric_aggregator.reset()
            describe_dataset(test_dataset)
            model.eval()
            test(model, test_loader, metric_aggregator)
            metric_aggregator.print_metrics()



if __name__ == '__main__':
    run_manual_camnet()