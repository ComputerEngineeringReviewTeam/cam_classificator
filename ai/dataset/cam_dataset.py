import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pandas import read_csv, DataFrame

from ai.dataset.image_desc import ImageFields


class CamDataset(Dataset):
    """
    Dataset for the CAM data

    Data structure:
        - csv_file: Contains the labels for the images
        - img_dir: Contains the images

    Args:
        csv_file: Path to the csv file containing the labels
        img_dir: Path to the directory containing the images
        transform: Transformation to be applied to the images

    __getitem__ returns:
        (transformed_image, scale), label
    """
    def __init__(self, csv_file: str, img_dir: str, transform=None):
        self.labels = read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data: DataFrame = self.labels.iloc[item]
        image_path = os.path.join(self.img_dir, str(data[ImageFields.ImageId]) + "." + data[ImageFields.PhotoType])
        image = read_image(image_path)
        label = torch.tensor([
                data[ImageFields.TotalArea],
                data[ImageFields.TotalLength],
                data[ImageFields.MeanThickness],
                data[ImageFields.BranchingPoints],
                data[ImageFields.IsGood],             # maybe handle this one differently
            ],
            dtype=torch.float32)
        scale = torch.tensor(data[ImageFields.Scale], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (image, scale), label


class StructuredCamDataset(CamDataset):
    """
    Dataset for structured data

    Data structure:
        - root_dir/             # Root directory
            - train.csv         # Contains the labels for the training data
            - train/            # Contains the training images
                - image1.jpg
                - image2.jpg
                - ...
            - test.csv          # Contains the labels for the testing data
            - test/             # Contains the testing images
                - image1.jpg
                - image2.jpg
                - ...

    Args:
        root_dir: Path to the directory containing the csv files and images directories
        transform: Transformation to be applied to the images
        train: Whether to load the training or testing data

    """
    TRAIN_DIR = "train/"
    TRAIN_CSV = "train.csv"
    TEST_DIR = "test/"
    TEST_CSV = "test.csv"

    def __init__(self, root_dir: str, transform=None, train: bool = True):
        if train:
            super().__init__(root_dir + self.TRAIN_CSV, root_dir + self.TRAIN_DIR, transform)
        else:
            super().__init__(root_dir + self.TEST_CSV, root_dir + self.TEST_DIR, transform)