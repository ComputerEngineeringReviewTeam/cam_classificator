import torch

from ai.utils.dataset_helpers import prepare_tensors

def train(model: torch.nn.Module,
          loss_fn: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str):
    """
    Trains a given model using a specified loss function, optimizer, and data loader over a defined number of epochs.
    Computes losses, performs backpropagation, and updates weights via the optimizer.

    Parameters:
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn (torch.nn.Module): The loss function used to compute the model's error.
        dataloader (torch.utils.data.DataLoader): DataLoader that supplies the training dataset in batches.
        optimizer (torch.optim.Optimizer): Optimization algorithm used to update the model's weights.
        epochs (int): The number of complete passes through the dataset during training.
        device (str): Hardware to be used for training ('cpu' or 'cuda').

    """
    print('Training...')
    for epoch in range(epochs):
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            binary_output, regression_output = model((image, scale))
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
    print('Finished Training')
