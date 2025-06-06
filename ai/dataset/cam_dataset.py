import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pandas import DataFrame

from ai.model.config import *
import utils.filters as filters
from ai.dataset.cam_label import LabelLoader, ColumnNames, CsvLabelLoader, JsonLabelLoader


COLUMNS_TO_NUM_LABELS = [ColumnNames.BranchingPoints]

def normalize_minmax(column_tensor, new_max=1.0, new_min=0.0):
    amin, amax = torch.amin(column_tensor), torch.amax(column_tensor)
    normalized_column_tensor = ((column_tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min
    return normalized_column_tensor


# TODO: remove scale
# TODO: make normalization optional
class CamDataset(Dataset):
    """
    Represents a custom dataset loader for managing labeled image data, allowing transformations,
    filters, normalization, and support for multiple label formats.

    The CamDataset class serves as a utility for handling labeled datasets of images. It supports
    normalizing label values, applying a series of filters to images for preprocessing, and loading
    from multiple label formats like CSV and JSON. This class is compatible with PyTorch's Dataset
    module, enabling seamless integration into data pipelines for training machine learning models.

    Attributes:
        img_dir: Directory path to the source images.
        transform: Transformation pipeline to be applied to images.
        imageFilterSet: Set of filters to process the images.
        dtype: Data type for tensors used in the dataset.
        labels: DataFrame containing image paths and associated labels.
        tensor_labels: Normalized labels prepared as tensors.

    Args:
        labels DataFrame: DataFrame containing image labels and paths.
        img_dir str: Directory where images are stored.
        transform Compose | None: Torchvision transformation applied to the image, optional.
        imageFilterSet flt.Filters | None: Filters applied to preprocess the image, optional.
        dtype: Data type for tensors, default is torch.float32.
        only_good bool: Flag to filter out images with "IsGood" label set to False, default is False.

    Methods:
        from_label_loader: Creates a CamDataset instance from a LabelLoader.
        from_csv: Creates a CamDataset instance by loading labels from a CSV file.
        from_json: Creates a CamDataset instance by loading labels from a JSON file.
        normalize_and_nanize_labels: Normalizes label columns and replaces specified values with NaN.
        __load_image: Internal method to load and preprocess an image.
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves the data and labels for a specific index in the dataset.

    """

    def __init__(self,
                 labels: DataFrame,
                 img_dir: str,
                 transform: Compose | None = None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,
                 only_good: bool = False):

        self.img_dir = img_dir
        self.transform = transform
        self.imageFilterSet = imageFilterSet
        self.dtype = dtype

        if only_good:
            self.labels = labels[labels[ColumnNames.IsGood] == True]
        else:
            self.labels = labels

        self.tensor_labels = self.normalize_and_nanize_labels(-1.0, COLUMNS_TO_NUM_LABELS)


    @classmethod
    def from_label_loader(cls,
                          label_loader: LabelLoader,
                          labels_path: str,
                          img_dir: str,
                          transform: Compose | None,
                          imageFilterSet: flt.Filters | None = None,
                          dtype=torch.float32,
                          only_good: bool = False):
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
        return cls(labels, img_dir, transform, imageFilterSet, dtype, only_good)

    @classmethod
    def from_csv(cls,
                 labels_path: str,
                 img_dir: str,
                 transform: Compose | None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,
                 only_good: bool = False):
        """
        Create a CamDataset from a csv file

        Args:
            labels_path: Path to the csv file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(CsvLabelLoader(), labels_path, img_dir, transform, imageFilterSet, dtype, only_good)

    @classmethod
    def from_json(cls,
                  labels_path: str,
                  img_dir: str,
                  transform: Compose | None,
                  imageFilterSet: flt.Filters | None = None,
                  dtype = torch.float32,
                  only_good: bool = False):
        """
        Create a CamDataset from a json file

        Args:
            labels_path: Path to the json file containing the labels
            img_dir: Path to the directory containing the images
            transform: Transformation to be applied to the images

        Returns:
            CamDataset object
        """
        return cls.from_label_loader(JsonLabelLoader(), labels_path, img_dir, transform, imageFilterSet, dtype, only_good)

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
            # normalized_column_tensor = column_tensor
            normalized_column_tensor[special_value_mask] = float('nan')
            labels_as_tensor.append(normalized_column_tensor)
        tensor_labels = torch.column_stack(labels_as_tensor)
        return tensor_labels

    def __load_image(self, image_name: str):
        """
        Loads image and applies filters.
        """
        image_path = os.path.join(self.img_dir, image_name)
        image = flt.Image(image_path)
        if DISPLAY_IMAGES_BEFORE_FILTERS:
            image.getImage().show()
        image = self.imageFilterSet.applyFilters(image)
        if DISPLAY_IMAGES_AFTER_FILTERS:
            image.getImage().show()
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        data = self.labels.iloc[item]

        image = self.__load_image(str(data[ColumnNames.ImageName]))
        image = image.getTensor() / 255.0   # most professional normalization; may be redundant if using transform Normalize()
        if self.transform:
            image = self.transform(image)

        scale = torch.tensor(data[ColumnNames.Scale], dtype=self.dtype)
        regression_target = self.tensor_labels[item]
        binary_target = torch.tensor(data[ColumnNames.IsGood], dtype=self.dtype)

        return (image, scale), (binary_target, regression_target)


