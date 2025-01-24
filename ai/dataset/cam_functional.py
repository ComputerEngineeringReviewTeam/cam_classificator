from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from ai.dataset.cam_dataset import CamDataset

def train_test_datasets(all_labels: DataFrame, img_dir: str, train_fraction: float, transform=None) -> tuple[CamDataset, CamDataset]:
    train_dataset = CamDataset(all_labels[:int(train_fraction * len(all_labels))], img_dir, transform)
    test_dataset = CamDataset(all_labels[int(train_fraction * len(all_labels)):], img_dir, transform)
    return train_dataset, test_dataset



def train_test_dataloaders(all_labels: DataFrame, img_dir: str, train_fraction: float, batch_size: int, shuffle: bool = False,
                           transform=None) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = train_test_datasets(all_labels, img_dir, train_fraction, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader