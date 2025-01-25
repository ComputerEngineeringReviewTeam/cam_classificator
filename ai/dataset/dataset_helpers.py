from pandas import DataFrame

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
