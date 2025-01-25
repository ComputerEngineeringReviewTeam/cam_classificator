import pandas as pd
import json
from abc import ABC, abstractmethod


class ColumnNames:
    """
    Column names to use for labeling the data in DataFrames
    """
    ImageId = "id"                        # use for index column
    TotalArea: str = "total_area"
    TotalLength: str = "total_length"
    MeanThickness: str = "mean_thickness"
    BranchingPoints: str = "branching_points"
    IsGood: str = "is_good"
    Scale: str = "scale"
    PhotoType: str = "photo_type"
    ImageName: str = "img"


class LabelLoader(ABC):
    """
    Abstract class for loading labels for CAM data
    """
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """
        Load the labels from the given path and return a DataFrame

        Args:
            path: Path to the file containing the labels

        Returns:
            DataFrame containing the labels
        """
        pass


class JsonLabelLoader(LabelLoader):
    """
    Class for loading labels from a json file
    """
    def load(self, path: str) -> pd.DataFrame:
        """
        Load the labels from the given json file and return a DataFrame

        Args:
            path: Path to the json file containing the labels

        Returns:
            DataFrame containing the labels
        """
        with open(path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame.from_dict(data, orient="index")
        return df


class CsvLabelLoader(LabelLoader):
    """
    Class for loading labels from a csv file
    """
    def load(self, path: str) -> pd.DataFrame:
        """
        Load the labels from the given csv file and return a DataFrame

        Args:
            path: Path to the csv file containing the labels

        Returns:
            DataFrame containing the labels
        """
        return pd.read_csv(path, index_col=ColumnNames.ImageId)