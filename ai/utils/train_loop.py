import torch

from ai.model.camnet_base import CamNetBase
from ai.loss_fn.camloss_base import CamLossBase
from ai.utils.dataset_helpers import prepare_tensors

def train(model: CamNetBase,
          loss_fn: CamLossBase,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str):
    """
    Trains a given model using a specified loss function, data loader, optimizer,
    number of epochs, and computational device.

    This function iteratively processes batches of data over multiple epochs,
    computes the loss, and adjusts the model weights using backpropagation and
    the specified optimizer. It provides updates about the training process
    through printed progress messages.

    Parameters:
    model : CamNetBas
        The model to be evaluated. It should be a subclass of the
        `ai.model.camnet_base.CamNetBase` class.
    loss_fn : CamLossBase
        The loss function used to compute the loss between the model outputs
        and the targets.
    dataloader : torch.utils.data.DataLoader
        An iterable that provides batches of data and corresponding targets
        for the training process.
    optimizer : torch.optim.Optimizer
        The optimization algorithm used to adjust the model's parameters based
        on computed gradients.
    epochs : int
        The number of complete passes through the dataset for training.
    device : str
        Specifies the computational device to be used for training, such as
        "cuda" for GPU or "cpu".

    Returns:
    None

    Raises:
    None
    """

    print('Training...')
    for epoch in range(epochs):
        for i, (image, (binary_target, regression_target)) in enumerate(dataloader):
            image, binary_target, regression_target = prepare_tensors(image, binary_target, regression_target, device)
            binary_output, regression_output = model(image)
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
    print('Finished Training')
