import torch
import abc
import PIL.Image


class CamNetBase(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, image: torch.Tensor | PIL.Image.Image) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Abstract method representing the forward pass for processing an image. This
        method must be implemented in subclasses, allowing for the computation of
        desired outputs from the input image.

        Parameters:
            image: torch.Tensor | PIL.Image.Image
                The input image for processing. This can either be a PyTorch tensor
                or an image represented by PIL.Image.

        Returns:
            tuple[torch.Tensor, torch.Tensor]
                The first element represents the prediction for binary classification of the tissue fitness
                and the second the prediction for numerical parameters. Any element may be None if it is not
                produced by the specific CamNet variant.
        """
        raise NotImplementedError