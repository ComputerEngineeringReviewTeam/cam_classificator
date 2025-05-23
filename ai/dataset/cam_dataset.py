import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import Compose, functional
from pandas import DataFrame
from PIL import Image, ImageFilter
from ai.model.config import *

from ai.dataset.cam_label import LabelLoader, ColumnNames, CsvLabelLoader, JsonLabelLoader
from ai.model.config import TARGET_SIZE

import utils.filters as filters

COLUMNS_TO_NORMALIZE = [ColumnNames.BranchingPoints]


def normalize_minmax(column_tensor, new_max = 1.0, new_min = 0.0):
    amin, amax = torch.amin(column_tensor), torch.amax(column_tensor)
    normalized_column_tensor = ((column_tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min
    return normalized_column_tensor


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
                 transform: Compose | None,
                 imageFilterSet: flt.Filters):
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform
        self.imageFilterSet = imageFilterSet
        self.tensor_labels = self.prep()

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

    def prep(self):
        labels_as_tensor = []
        for column_name in COLUMNS_TO_NORMALIZE:
            column_tensor = torch.tensor(self.labels[column_name].values, dtype=torch.float32)
            special_value_mask = column_tensor == -1.0
            normalized_column_tensor = normalize_minmax(column_tensor)
            normalized_column_tensor[special_value_mask] = float('nan')
            labels_as_tensor.append(normalized_column_tensor)
        tensor_labels = torch.column_stack(labels_as_tensor)
        return tensor_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        data = self.labels.iloc[item]

        image_path = os.path.join(self.img_dir, str(data[ColumnNames.ImageName]))
        image = flt.Image(image_path)
        if DISPLAY_IMAGES_BEFORE_FILTERS:
            image.getImage().show()
        image = self.imageFilterSet.applyFilters(image)
        if DISPLAY_IMAGES_AFTER_FILTERS:
            image.getImage().show()

        image = image.getTensor() / 255.0   # most professional normalisation
        scale = torch.tensor(data[ColumnNames.Scale], dtype=torch.float32)
        regression_target = self.tensor_labels[item]
        binary_target = torch.tensor(data[ColumnNames.IsGood], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (image, scale), (binary_target, regression_target)


