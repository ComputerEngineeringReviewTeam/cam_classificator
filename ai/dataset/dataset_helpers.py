import pandas as pd
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ai.dataset.cam_dataset import CamDataset
from ai.dataset.cam_label import ColumnNames


def train_test_sample(all_labels: DataFrame,
                      train_fraction: float,
                      random_state: int | None = None) -> tuple[DataFrame, DataFrame]:
    """
    Splits the labels into train and test samples according to the train_fraction

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        train_fraction: Fraction of the data to be used for training
        random_state: Random seed for reproducibility

    Returns:
        train_data, test_data
    """

    train_data = all_labels.sample(frac=train_fraction, random_state=random_state)
    test_data = all_labels.drop(train_data.index)
    return train_data, test_data


def train_test_balanced_sample(all_labels: DataFrame,
                               train_fraction: float,
                               random_state: int | None = None) -> tuple[DataFrame, DataFrame]:
     """
     Splits the labels into train and test samples according to the train_fraction, ensuring that the classes are balanced

     Args:
          all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
          train_fraction: Fraction of the data to be used for training
          random_state: Random seed for reproducibility

     Returns:
          train_data, test_data
     """

     good_labels = all_labels[all_labels[ColumnNames.IsGood] == 1]
     bad_labels = all_labels[all_labels[ColumnNames.IsGood] == 0]

     good_labels_for_train = good_labels.sample(frac=train_fraction, random_state=random_state)
     good_labels_for_test = good_labels.drop(good_labels_for_train.index)

     bad_labels_for_train = bad_labels.sample(frac=train_fraction, random_state=random_state)
     bad_labels_for_test = bad_labels.drop(bad_labels_for_train.index)

     train_data = pd.concat([good_labels_for_train, bad_labels_for_train])
     test_data = pd.concat([good_labels_for_test, bad_labels_for_test])

     return train_data, test_data


def sample_labels(all_labels: DataFrame,
                  train_fraction: float,
                  random_state: int | None = None,
                  balanced: bool = False) -> tuple[DataFrame, DataFrame]:
    """
    Splits the labels into train and test samples according to the train_fraction,
    can also ensure that the classes are balanced

    Wrapper for train_test_sample and train_test_balanced_sample

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        train_fraction: Fraction of the data to be used for training
        random_state: Random seed for reproducibility
        balanced: bool, if True the classes will be balanced

    Returns:
        train_data, test_data
    """
    if balanced:
        return train_test_balanced_sample(all_labels, train_fraction, random_state)
    return train_test_sample(all_labels, train_fraction, random_state)

def train_test_datasets(all_labels: DataFrame,
                        img_dir: str,
                        train_fraction: float,
                        transform: Compose | None,
                        random_state: int | None = None,
                        balanced: bool = False) -> tuple[CamDataset, CamDataset]:

    """
    Splits the labels into train and test datasets according to the train_fraction

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        img_dir: Path to the directory containing the images
        train_fraction: Fraction of the data to be used for training
        transform: Transformation to be applied to the images
        random_state: Random seed for reproducibility
        balanced: bool, if True the classes will be balanced

    Returns:
        train_dataset, test_dataset
    """

    train_data, test_data = sample_labels(all_labels, train_fraction, random_state, balanced)
    return CamDataset(train_data, img_dir, transform), CamDataset(test_data, img_dir, transform)


def to_dataloaders(train_dataset: CamDataset,
                   test_dataset: CamDataset,
                   batch_size: int) -> tuple[DataLoader, DataLoader]:

    """
    Creates dataloaders for the train and test datasets

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size

    Returns:
        train_dataloader, test_dataloader
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def train_test_data(all_labels: DataFrame,
                    img_dir: str,
                    train_fraction: float,
                    batch_size: int,
                    transform: Compose | None,
                    random_state: int | None = None,
                    balanced: bool = False)\

        -> tuple[tuple[CamDataset, CamDataset], tuple[DataLoader, DataLoader]]:
    """
    Creates train and test dataloaders from the labels

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        img_dir: Path to the directory containing the images
        train_fraction: Fraction of the data to be used for training
        batch_size: Batch size
        transform: Transformation to be applied to the images
        random_state: Random seed for reproducibility
        balanced: bool, if True the classes will be balanced

    Returns:
        train_dataloader, test_dataloader
    """
    train_dataset, test_dataset = train_test_datasets(all_labels, img_dir, train_fraction, transform, random_state, balanced)
    train_dataloader, test_dataloader = to_dataloaders(train_dataset, test_dataset, batch_size)

    return (train_dataset, test_dataset), (train_dataloader, test_dataloader)


def describe_dataset(dataset: CamDataset):
    """
    Describes the dataset by printing the counts of the unique values in the ColumnNames.IsGood column
    Args:
        dataset: torch.utils.data.Dataset
    """
    print("Describing dataset")
    print(dataset.labels[ColumnNames.IsGood].value_counts())
