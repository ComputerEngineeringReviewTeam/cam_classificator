import torch

from ai.metrics.metric_collector import CamMetricCollector
from ai.utils.dataset_helpers import prepare_tensors

def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         metrics: CamMetricCollector,
         device: str):
    """
    Tests a PyTorch model using data from a dataloader and collects metrics.

    Arguments:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        dataloader (torch.utils.data.DataLoader): An iterator that provides
            batches of data for testing, including input features and
            corresponding targets.
        metrics (CamMetricCollector): Object for collecting and
            aggregating metrics during the evaluation process.
        device (str): Device to transfer the tensors to, typically 'cpu'
            or 'cuda'.

    Note:
        This function runs in no gradient mode by default.
    """
    print("Testing...")
    with torch.no_grad():
        for i, (image, (binary_target, regression_target)) in enumerate(dataloader):
            image, binary_target, regression_target = prepare_tensors(image, binary_target, regression_target, device)
            binary_output, regression_output = model(image)

            metrics(binary_output, regression_output, binary_target, regression_target)
