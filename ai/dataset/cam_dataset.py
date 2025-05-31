import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pandas import DataFrame
from ai.model.config import *

import utils.filters as filters
from ai.dataset.cam_label import LabelLoader, ColumnNames, CsvLabelLoader, JsonLabelLoader
from ai.dataset.dataset_utils import normalize_minmax

COLUMNS_TO_NUM_LABELS = [ColumnNames.BranchingPoints]


class CamDataset(Dataset):
    """
    Dataset for the CAM data

    Args:
        labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        img_dir: Path to the directory containing the images
        transform: Transformation to be applied to the images

    __getitem__() returns:
        (transformed_image, scale), (binary_target, regression_target)
    """

    def __init__(self,
                 labels: DataFrame,
                 img_dir: str,
                 transform: Compose | None = None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,):
        self.labels = labels
        # self.labels = labels[labels[ColumnNames.IsGood] == True]
        self.img_dir = img_dir
        self.transform = transform
        self.imageFilterSet = imageFilterSet
        self.dtype = dtype
        self.tensor_labels = self.normalize_and_nanize_labels(-1.0, COLUMNS_TO_NUM_LABELS)


    @classmethod
    def from_label_loader(cls,
                          label_loader: LabelLoader,
                          labels_path: str,
                          img_dir: str,
                          transform: Compose | None):
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
        return cls(labels, img_dir, transform)

    @classmethod
    def from_csv(cls,
                 labels_path: str,
                 img_dir: str,
                 transform: Compose | None):
        """
        Create a CamDataset from a csv file

        Args:
            labels_path: Path to the csv file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(CsvLabelLoader(), labels_path, img_dir, transform)

    @classmethod
    def from_json(cls,
                  labels_path: str,
                  img_dir: str,
                  transform: Compose | None):
        """
        Create a CamDataset from a json file

        Args:
            labels_path: Path to the json file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(JsonLabelLoader(), labels_path, img_dir, transform)

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        data = self.labels.iloc[item]

        image = self.__load_image(str(data[ColumnNames.ImageName]))
        image = image.getTensor() / 255.0   # most professional normalization; may be redundant if doing
        if self.transform:
            image = self.transform(image)

        scale = torch.tensor(data[ColumnNames.Scale], dtype=self.dtype)
        regression_target = self.tensor_labels[item]
        binary_target = torch.tensor(data[ColumnNames.IsGood], dtype=self.dtype)

        return (image, scale), (binary_target, regression_target)

    def __load_image(self, image_name: str):
        image_path = os.path.join(self.img_dir, image_name)
        image = flt.Image(image_path)
        if DISPLAY_IMAGES_BEFORE_FILTERS:
            image.getImage().show()
        image = self.imageFilterSet.applyFilters(image)
        if DISPLAY_IMAGES_AFTER_FILTERS:
            image.getImage().show()
        return image


