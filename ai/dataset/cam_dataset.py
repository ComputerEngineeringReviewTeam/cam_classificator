import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pandas import DataFrame
import warnings

from ai.config import DISPLAY_IMAGES_AFTER_FILTERS, DISPLAY_IMAGES_BEFORE_FILTERS   # TODO: decouple from config
from ai.dataset.cam_label import LabelLoader, ColumnNames, CsvLabelLoader, JsonLabelLoader
import ai.utils.filters as flt


COLUMNS_TO_NORMALIZE = [ColumnNames.BranchingPoints]


def normalize_minmax(tensor,
                     amin: float | None = None,
                     amax: float | None = None,
                     new_max=1.0,
                     new_min=0.0):
    """
    Normalize a tensor using min-max scaling.

    This function scales a PyTorch tensor to a specific range [new_min, new_max]
    using the min-max normalization technique. The minimum (amin) and maximum (amax)
    can either be provided or calculated directly from the tensor. This method is
    commonly used for feature scaling in preprocessing steps to bring the data into
    a comparable range for machine learning tasks.

    Args:
        tensor: A PyTorch tensor to be normalized.
        amin: Optional value for the minimum of the tensor's range. If not provided,
            it will be calculated using the tensor's minimum value.
        amax: Optional value for the maximum of the tensor's range. If not provided,
            it will be calculated using the tensor's maximum value.
        new_max: The maximum value of the scaled range. Defaults to 1.0.
        new_min: The minimum value of the scaled range. Defaults to 0.0.

    Returns:
        The normalized tensor where values are scaled to the range specified
        by [new_min, new_max].
    """
    if amin is None:
        amin = torch.amin(tensor)
    if amax is None:
        amax = torch.amax(tensor)

    normalized_column_tensor = ((tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min
    return normalized_column_tensor


# TODO: make normalization optional
class CamDataset(Dataset):


    def __init__(self,
                 labels: DataFrame,
                 img_dir: str,
                 transform: Compose | None = None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,
                 only_good = False,
                 normalize = True):

        self.img_dir = img_dir
        self.transform = transform
        self.imageFilterSet = imageFilterSet
        self.dtype = dtype
        self.labels = labels

        if only_good:
            self.labels = labels[labels[ColumnNames.IsGood] == True]

        if normalize:
            self.labels[COLUMNS_TO_NORMALIZE] = normalize_minmax(self.labels[COLUMNS_TO_NORMALIZE], 0, 200)

    @classmethod
    def from_label_loader(cls,
                          label_loader: LabelLoader,
                          labels_path: str,
                          img_dir: str,
                          transform: Compose | None,
                          imageFilterSet: flt.Filters | None = None,
                          dtype = torch.float32,
                          only_good = False,
                          normalize = True):
        """
        Create a CamDataset from a LabelLoader

        Args:
            label_loader: LabelLoader object
            labels_path: Path to the file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        labels = label_loader.load(labels_path)
        return cls(labels, img_dir, transform, imageFilterSet, dtype, only_good, normalize)

    @classmethod
    def from_csv(cls,
                 labels_path: str,
                 img_dir: str,
                 transform: Compose | None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,
                 only_good = False,
                 normalize = True):
        """
        Create a CamDataset from a csv file

        Args:
            labels_path: Path to the csv file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(CsvLabelLoader(), labels_path, img_dir,
                                     transform, imageFilterSet, dtype, only_good, normalize)

    @classmethod
    def from_json(cls,
                  labels_path: str,
                  img_dir: str,
                  transform: Compose | None,
                  imageFilterSet: flt.Filters | None = None,
                  dtype = torch.float32,
                  only_good = False,
                  normalize = True):
        """
        Create a CamDataset from a json file

        Args:
            labels_path: Path to the json file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(JsonLabelLoader(), labels_path, img_dir,
                                     transform, imageFilterSet, dtype, only_good, normalize)

    @warnings.deprecated
    def normalize_and_nanize_labels(self, value_to_nan: any, column_list: list[str]):
        """
        Prepares and processes label data into a tensor format by normalizing values
        within each column and replacing special values with NaN. The processed data
        is stacked column-wise to create a composite tensor representation for use
        in further computation.

        Args:
            value_to_nan: special value that will be replaced by NaN values
            column_list: list of column names to process from the labels dataframe

        Returns:
            Tensor: A 2D tensor where each column corresponds to labeled data
            normalized using the min-max approach, with special values replaced
            by NaN.
        """
        labels_as_tensor = []
        for column_name in column_list:
            column_tensor = torch.tensor(self.labels[column_name].values, dtype=self.dtype)
            special_value_mask = column_tensor == value_to_nan
            normalized_column_tensor = normalize_minmax(column_tensor)
            normalized_column_tensor[special_value_mask] = float('nan')
            labels_as_tensor.append(normalized_column_tensor)
        tensor_labels = torch.column_stack(labels_as_tensor)
        return tensor_labels

    # TODO: decouple from config
    def __load_image(self, image_name: str) -> flt.Image:
        """
        Loads image and applies filters.
        """
        image_path = os.path.join(self.img_dir, image_name)
        image = flt.Image.fromPath(image_path)
        if DISPLAY_IMAGES_BEFORE_FILTERS:
            image.getImage().show()
        image = self.imageFilterSet.applyFilters(image)
        if DISPLAY_IMAGES_AFTER_FILTERS:
            image.getImage().show()
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Lazily loads the image and applies the configured filters and transforms.
        Returns the label data split into binary classification label ant other numerical parameters.

        Args:
            item: Index of the image and target to be retrieved

        Returns:
            - 3D torch.Tensor representing the preprocessed image
                - 1D torch.Tensor with the IsGood label in dtype
                - 1D torch.Tensor with the BranchingPoints label in dtype
        """
        data = self.labels.iloc[item]

        image = self.__load_image(str(data[ColumnNames.ImageName]))

        if self.transform:
            image = self.transform(image.getImage())
        else:
            image = image.getTensor()

        regression_target = data[ColumnNames.BranchingPoints]
        binary_target = torch.tensor(data[ColumnNames.IsGood], dtype=self.dtype)

        return image, (binary_target, regression_target)
