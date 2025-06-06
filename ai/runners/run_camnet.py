import torch
import torchmetrics as metrics

import ai.config as conf
from ai.dataset.cam_dataset import CamDataset
from ai.dataset.cam_label import JsonLabelLoader
from ai.model.camnet_regressor import CamNetRegressor
from ai.loss_fn.camloss_regressor import CamLossRegressor
from ai.utils.dataset_helpers import to_dataloaders, describe_dataset, sample_labels
from ai.utils.metrics import *
from ai.utils.train_loop import train
from ai.utils.test_loop import test
from ai.utils.transforms import CamTransforms


def run_camnet():
    # Transforms for the images used
    train_tsfms = CamTransforms.Train.grayscale
    test_tsfms = CamTransforms.Test.grayscale

    # Create datasets and dataloaders for loading the data using shortcut functions
    data_df = JsonLabelLoader().load(conf.LABELS_PATH)
    train_data, test_data = sample_labels(data_df, conf.TRAIN_FRACTION, conf.RANDOM_SEED, balanced=conf.BALANCED_SPLIT)
    train_dataset = CamDataset(train_data, conf.IMG_DIR, train_tsfms, conf.datasetFilterSet)
    test_dataset = CamDataset(test_data, conf.IMG_DIR, test_tsfms, conf.datasetFilterSet)
    train_loader, test_loader = to_dataloaders(train_dataset, test_dataset, batch_size=conf.BATCH_SIZE)

    # Create the model, optimizer and loss function
    model = CamNetRegressor(model_name=conf.MODEL_NAME,
                            pretrained=True,
                            num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    loss_fn = CamLossRegressor()

    # Create the metrics used for testing the model
    metric_aggregator = CamMetricCollector(
        classification_metrics=[
            BinaryMetrics(),
            metrics.classification.BinaryAccuracy(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            metrics.classification.BinaryRecall(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            metrics.classification.BinaryPrecision(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
        ], regression_metrics=[
            metrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            metrics.regression.MeanAbsoluteError().to(device=conf.DEVICE),  # average error
            metrics.regression.NormalizedRootMeanSquaredError().to(device=conf.DEVICE),  # standard deviation
            metrics.regression.R2Score().to(device=conf.DEVICE),  # ~accuracy
            RelativeError(),
            AverageRelativeError()
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
            model = CamNetRegressor(model_name=conf.MODEL_NAME,
                                    pretrained=True,
                                    num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
            model.load_state_dict(torch.load(conf.MODEL_PATH, weights_only=True))
        model.eval()
        test(model, test_loader, metric_aggregator)
        metric_aggregator.print_metrics()


if __name__ == '__main__':
    run_camnet()
