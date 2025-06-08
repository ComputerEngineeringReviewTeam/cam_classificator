import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from pandas import DataFrame

from ai.config import *
from ai.dataset.cam_label import LabelLoader, ColumnNames, CsvLabelLoader, JsonLabelLoader


# TODO: normalization 0-200
# TODO: all photos
def normalize_minmax(column_tensor, new_max=1.0, new_min=0.0):
    # amin, amax = torch.amin(column_tensor), torch.amax(column_tensor)
    amin, amax = 0.0, 200.0
    normalized_column_tensor = ((column_tensor - amin) / (amax - amin)) * (new_max - new_min) + new_min
    return normalized_column_tensor


# TODO: remove scale
# TODO: make normalization optional
class CamBinnedDataset(Dataset):
    """
    A dataset class for handling image data associated with labels and bin-based
    classification for use in machine learning tasks.

    This class is designed to preprocess and manage image data alongside
    label information, supporting various format loaders (e.g., CSV, JSON).
    It also enables normalization of labels and filtering of images based on
    custom filters. It is suitable for tasks requiring classification into bins,
    binary target generation, and optional transformations. One can use this
    class with different file formats and perform preprocessing tasks seamlessly.

    Attributes:
        img_dir (str): Directory path where images are stored.
        transform (Compose | None): Transformation object to apply to images.
        imageFilterSet (flt.Filters | None): Set of filters for preprocessing images.
        dtype: Data type for tensors, defaults to torch.float32.
        bins (torch.Tensor): Bin boundaries for classifying labels.
        class_probabilities (bool): Flag indicating whether to return class
            probabilities or class indices.
        labels (DataFrame): DataFrame holding all label information associated with images.
        tensor_labels: Tensor representation of a specific label column.
    """

    def __init__(self,
                 labels: DataFrame,
                 img_dir: str,
                 bins: torch.Tensor,
                 transform: Compose | None = None,
                 imageFilterSet: flt.Filters | None = None,
                 dtype = torch.float32,
                 only_good: bool = False,
                 class_probabilities: bool = False):

        self.img_dir = img_dir
        self.transform = transform
        self.imageFilterSet = imageFilterSet
        self.dtype = dtype
        self.bins = bins
        self.class_probabilities = class_probabilities

        if only_good:
            self.labels = labels[labels[ColumnNames.IsGood] == True]
        else:
            self.labels = labels

        # self.labels = labels[labels[ColumnNames.BranchingPoints] <= 15]

        self.tensor_labels = self.labels[ColumnNames.BranchingPoints]

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

    def bins_num(self):
        """
        Returns the number of bins (classes) in the dataset.
        Note that torch.bucketize() is used internally, which creates bins differently than numpy.digitize() or panadas.cut().
        With bins=[1, 5] the bins will be (-inf, 1], (1, 5], (5, inf)
        """
        return self.bins.size(dim=0) + 1

    def __getitem__(self, item) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        data = self.labels.iloc[item]

        image = self.__load_image(str(data[ColumnNames.ImageName]))
        image = image.getTensor() / 255.0   # most professional normalization; may be redundant if using transform Normalize()
        if self.transform:
            image = self.transform(image)

        scale = torch.tensor(data[ColumnNames.Scale], dtype=self.dtype)

        class_id = torch.bucketize(data[ColumnNames.BranchingPoints], boundaries=self.bins)
        if self.class_probabilities:
            regression_target = torch.zeros(self.bins_num())
            regression_target[class_id] = 1.0
        else:
            regression_target = torch.tensor(class_id, dtype=torch.long)

        binary_target = torch.tensor(data[ColumnNames.IsGood], dtype=self.dtype)

        return (image, scale), (binary_target, regression_target)
