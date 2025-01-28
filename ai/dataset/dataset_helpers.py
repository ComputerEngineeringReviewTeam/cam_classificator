from pandas import DataFrame
from torch.utils.data import DataLoader

from ai.dataset.cam_dataset import CamDataset

def train_test_datasets(all_labels: DataFrame, img_dir: str, train_fraction: float, transform=None) -> tuple[CamDataset, CamDataset]:
    """
    Splits the labels into train and test datasets according to the train_fraction

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        img_dir: Path to the directory containing the images
        train_fraction: Fraction of the data to be used for training
        transform: Transformation to be applied to the images

    Returns:
        train_dataset, test_dataset
    """
    train_dataset = CamDataset(all_labels[:int(train_fraction * len(all_labels))], img_dir, transform)
    test_dataset = CamDataset(all_labels[int(train_fraction * len(all_labels)):], img_dir, transform)
    return train_dataset, test_dataset


def get_dataloaders(train_dataset: CamDataset, test_dataset: CamDataset, batch_size: int) -> tuple[DataLoader, DataLoader]:
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


def train_test_data(all_labels: DataFrame, img_dir: str, train_fraction: float, batch_size: int, transform=None)\
        -> tuple[tuple[CamDataset, CamDataset], tuple[DataLoader, DataLoader]]:
    """
    Creates train and test dataloaders from the labels

    Args:
        all_labels: DataFrame containing the labels, columns should be the ones defined in cam_label.ColumnNames
        img_dir: Path to the directory containing the images
        train_fraction: Fraction of the data to be used for training
        batch_size: Batch size
        transform: Transformation to be applied to the images

    Returns:
        train_dataloader, test_dataloader
    """
    train_dataset, test_dataset = train_test_datasets(all_labels, img_dir, train_fraction, transform)
    train_dataloader, test_dataloader = get_dataloaders(train_dataset, test_dataset, batch_size)
    return (train_dataset, test_dataset), (train_dataloader, test_dataloader)


def describe_dataset(dataset: CamDataset):
    """
    Describes the dataset by printing the counts of the unique values in the 'is_good' column

    Args:
        dataset: torch.utils.data.Dataset
    """
    print("Describing dataset")
    print(dataset.labels['is_good'].value_counts())
