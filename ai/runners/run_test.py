import torchmetrics as metrics

import ai.config as conf
from ai.dataset.cam_dataset import CamDataset
from ai.model import CamNet, CamNetConv
from ai.utils.dataset_helpers import describe_dataset
from ai.metrics.metrics import *
from ai.utils.test_loop import test
from ai.utils.transforms import CamTransforms
from ai.metrics.metrics_colls import MetricsCollections


def run_camnet():
    # Transforms for the images used
    test_tsfms = conf.TEST_TF

    # Create datasets and dataloaders for loading the data using shortcut functions
    test_dataset = CamDataset.from_json(labels_path=conf.LABELS_PATH,
                                        img_dir=conf.IMG_DIR,
                                        transform=test_tsfms,
                                        imageFilterSet=conf.datasetFilterSet,
                                        only_good=conf.ONLY_GOOD)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.BATCH_SIZE)

    # Create the model
    model = CamNet(model_name=conf.MODEL_NAME,
                   mode=conf.MODE,
                   pretrained=True,
                   num_aux_inputs=conf.NUM_AUX_INPUTS).to(device=conf.DEVICE)
    model.load_state_dict(torch.load(conf.MODEL_PATH, weights_only=True))

    # Create the metrics used for testing the model
    metric_aggregator = MetricsCollections.std

    # Run testing loop
    describe_dataset(test_dataset)
    model.eval()
    test(model, test_loader, metric_aggregator)
    metric_aggregator.print_metrics()


if __name__ == '__main__':
    run_camnet()
