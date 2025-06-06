import torch

import ai.config as conf
from ai.utils.metrics import CamMetricCollector
from ai.utils.dataset_helpers import prepare_tensors

def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         metric_aggregator: CamMetricCollector,
         device: str = conf.DEVICE):
    with torch.no_grad():
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            binary_output, regression_output = model((image, scale))

            metric_aggregator(binary_output, regression_output, binary_target, regression_target)