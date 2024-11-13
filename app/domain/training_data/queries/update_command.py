"""
About: A command for updating training data
Author: Paweł Bogdanowicz
"""

from bson import ObjectId


class UpdateTrainingDataCommand:
    def __init__(self):
        self._id: ObjectId|None = None
        self._branching_points: float|None = None
        self._total_length: float|None = None
        self._mean_thickness: float|None = None
        self._total_area: float|None = None

    def to_mongo_json(self) -> dict:
        return {
            'branching_points': self._branching_points,
            'total_length': self._total_length,
            'mean_thickness': self._mean_thickness,
            'total_area': self._total_area
        }

    def set_id(self, id: ObjectId):
        self._id = id
        return self

    def get_id(self):
        return self._id

    def set_branching_points(self, branching_points: float):
        self._branching_points = branching_points
        return self

    def set_total_length(self, total_length: float):
        self._total_length = total_length
        return self

    def set_mean_thickness(self, mean_thickness: float):
        self._mean_thickness = mean_thickness
        return self

    def set_total_area(self, total_area: float):
        self._total_area = total_area
        return self