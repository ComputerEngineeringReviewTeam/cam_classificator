import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from pandas import read_csv, DataFrame


class CamDataset(Dataset):
    def __init__(self, labels_file_path: str, img_dir: str, transform=None):
        self.labels = read_csv(labels_file_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data: DataFrame = self.labels.iloc[item]
        image_path = os.path.join(self.img_dir, str(data['id']) + "." + data['photo_type'])
        image = read_image(image_path)
        label = torch.tensor([
                data['total_area'],
                data['total_length'],
                data['mean_thickness'],
                data['branching_points'],
                data['is_good'],             # maybe handle this one differently
            ],
            dtype=torch.float32)
        scale = torch.tensor(data['scale'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return (image, scale), label