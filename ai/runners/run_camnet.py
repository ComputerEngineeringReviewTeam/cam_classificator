import torch

import global_config as gconf
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


def run_camnet():
    # Transforms for the images used
    train_tsfms = CamTransforms.Train.grayscale
    test_tsfms = CamTransforms.Test.grayscale

    # Create datasets and dataloaders for loading the data using shortcut functions
    data_df = JsonLabelLoader().load(gconf.LABELS_PATH)
    train_data, test_data = sample_labels(data_df, conf.TRAIN_FRACTION, conf.RANDOM_SEED, balanced=conf.BALANCED_SPLIT)
    train_dataset = CamDataset(train_data, gconf.IMG_DIR, train_tsfms, conf.datasetFilterSet, only_good=conf.ONLY_GOOD)
    test_dataset = CamDataset(test_data, gconf.IMG_DIR, test_tsfms, conf.datasetFilterSet, only_good=conf.ONLY_GOOD)
    train_loader, test_loader = to_dataloaders(train_dataset, test_dataset, batch_size=conf.BATCH_SIZE)

    # Create the model, optimizer and loss function
    model = CamNet(model_name=conf.MODEL_NAME,
                   mode=conf.MODE,
                   pretrained=True,
                   num_aux_inputs=conf.NUM_AUX_INPUTS,
                   features=conf.FEATURES,
                   dropout=conf.DROPOUT).to(device=conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)
    loss_fn = CamLoss(mode=conf.MODE)

    # Create the metrics used for testing the model
    metric_aggregator = MetricsCollections.only_regressor

    # Run training loop
    if conf.TRAIN:
        describe_dataset(train_dataset)
        model.train()
        train(model, loss_fn, train_loader, optimizer, conf.EPOCHS, conf.DEVICE)
        if conf.SAVE_MODEL:
            torch.save(model.state_dict(), gconf.SAVED_MODEL_PATH)

    # Run testing loop
    if conf.TEST:
        describe_dataset(test_dataset)
        if conf.LOAD_MODEL:
            model = CamNet(model_name=conf.MODEL_NAME,
                           mode=conf.MODE,
                           pretrained=True,
                           num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
            model.load_state_dict(torch.load(gconf.SAVED_MODEL_PATH, weights_only=True))
        model.eval()
        test(model, test_loader, metric_aggregator, conf.DEVICE)
        metric_aggregator.print_metrics()


if __name__ == '__main__':
    run_camnet()
