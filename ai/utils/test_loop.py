import torch

from ai.model.camnet_base import CamNetBase
from ai.metrics.metric_collector import CamMetricCollector
from ai.utils.dataset_helpers import prepare_tensors

def test(model: CamNetBase,
         dataloader: torch.utils.data.DataLoader,
         metrics: CamMetricCollector,
         device: str):
    """
    Test a provided model using data from a dataloader and compute metrics. This function
    processes batches of data, applies the model to generate outputs, and evaluates those
    outputs against the targets using the provided metrics object. It operates in inference
    mode, ensuring that no gradients are computed.

    Arguments:
        model (CamNetBase): The model to be evaluated. It should be a subclass of the
            `ai.model.camnet_base.CamNetBase` class.
        dataloader (torch.utils.data.DataLoader): A DataLoader object providing the test dataset
            in batches.
        metrics (CamMetricCollector): An object for collecting and computing evaluation metrics.
            It should be callable and able to handle the model's outputs and corresponding
            targets.
        device (str): The device on which the computation will take place (e.g., 'cpu' or 'cuda').

    Raises:
        None

    Returns:
        None
    """

    print("Testing...")
    with torch.no_grad():
        for i, (image, (binary_target, regression_target)) in enumerate(dataloader):
            image, binary_target, regression_target = prepare_tensors(image, binary_target, regression_target, device)
            binary_output, regression_output = model(image)

            metrics(binary_output, regression_output, binary_target, regression_target)
