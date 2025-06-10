import torch
import abc


class CamLossBase(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self,
                binary_output: torch.Tensor | None,
                regression_output: torch.Tensor | None,
                binary_target: torch.Tensor | None,
                regression_target: torch.Tensor | None) -> torch.Tensor:
        """
        An abstract method that must be implemented by subclasses to define the
        specific forward pass for computing loss from binary and regression outputs and targets.
        Does not have to use all the tensors, but should return singular loss value.

        Parameters:
            binary_output (torch.Tensor): The predicted binary output tensor from the model.
            regression_output (torch.Tensor): The predicted regression output tensor from the model.
            binary_target (torch.Tensor): The ground truth binary target tensor.
            regression_target (torch.Tensor): The ground truth regression target tensor.

        Returns:
            torch.Tensor: Combined loss tensor.
        """
        raise NotImplementedError