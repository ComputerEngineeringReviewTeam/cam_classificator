import PIL.Image
import torch
import torchvision.transforms
from typing import Callable
import os

from ..prep.filters import Filters, Image


class ModelWrapper:
    def __init__(self,
                 model: torch.nn.Module,
                 filters: Filters | None = None,
                 transforms: torchvision.transforms.Compose | None = None,
                 interpret_fn: Callable | None = None):
        self.model = model.eval()
        self.filters = filters
        self.transforms = transforms
        self.interpret_fn = interpret_fn

    def preprocess(self, image: PIL.Image.Image) -> torch.Tensor | Image:
        """Run the filters and transforms the image"""
        # Convert to flt.Image to be able to apply filters
        image = Image(image)
        if self.filters is not None:
            image = self.filters.applyFilters(image)

        # Apply transforms on PIL.Image
        # NOTE: normalization should be done through torchvision.transforms.ToTensor()
        if self.transforms is not None:
            image = self.transforms(image.getImage())

        # Unsqueeze to add dim
        image = image.unsqueeze(0)
        return image

    def predict(self, image: torch.Tensor | PIL.Image.Image) -> torch.Tensor:
        """Run inference on the given preprocessed image"""
        with torch.no_grad():
            return self.model(image)

    def __call__(self, image: PIL.Image.Image):
        """Preprocess the image, run inference and return an interpreted result"""
        image = self.preprocess(image)
        output = self.predict(image)
        if self.interpret_fn is not None:
            output = self.interpret_fn(output)
        return output


def wrap_model(model: torch.nn.Module,
               state_dict_path: str | os.PathLike,
               filters: Filters | None = None,
               transforms: torchvision.transforms.Compose | None = None,
               interpret_fn: Callable | None = None
               ) -> ModelWrapper:
    model.load_state_dict(torch.load(state_dict_path,
                                     weights_only=True,
                                     map_location=torch.device('cpu')))
    model.eval()
    return ModelWrapper(model, filters, transforms, interpret_fn)