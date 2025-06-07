import torch

import ai.config as conf
from ai.utils.dataset_helpers import prepare_tensors

def train(model: torch.nn.Module,
          loss_fn: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int = conf.EPOCHS,
          device: str = conf.DEVICE):
    print('Training...')
    for epoch in range(epochs):
        for i, ((image, scale), (binary_target, regression_target)) in enumerate(dataloader):
            image, scale, binary_target, regression_target = prepare_tensors(image, scale, binary_target,
                                                                             regression_target, device)
            binary_output, regression_output = model((image, scale))  # model.random_trans(image)
            loss = loss_fn(binary_output, regression_output, binary_target, regression_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} Loss: {loss.item()}")
    print('Finished Training')