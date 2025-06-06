import os.path
import torch
import torchmetrics as metrics

import ai.config as conf
from ai.dataset.cam_dataset import CamDataset
from ai.dataset.cam_label import JsonLabelLoader
from ai.model import CamNet, CamNetConv
from ai.loss_fn import CamLoss
from ai.utils.dataset_helpers import to_dataloaders, describe_dataset, sample_labels
from ai.utils.train_loop import train
from ai.utils.test_loop import test
from ai.utils.transforms import CamTransforms
from ai.metrics.metrics_colls import MetricsCollections


def run_manual_camnet():
    MODELS_DIR = os.path.join(conf.CAM_ROOT, "ai", "saved_models")

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
    model = CamNet(model_name=conf.MODEL_NAME,
                   mode=conf.MODE,
                   pretrained=True,
                   num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    loss_fn = CamLoss(mode=conf.MODE)

    # Create the metrics used for testing the model
    metric_aggregator = MetricsCollections.std

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
                new_model = CamNet(model_name=conf.MODEL_NAME,
                               mode=conf.MODE,
                               pretrained=True,
                               num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
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
            describe_dataset(test_dataset)
            model.eval()
            test(model, test_loader, metric_aggregator)
            metric_aggregator.print_metrics()
            metric_aggregator.reset()


if __name__ == '__main__':
    run_manual_camnet()