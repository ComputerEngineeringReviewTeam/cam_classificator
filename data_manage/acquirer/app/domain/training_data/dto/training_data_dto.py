"""
About: A DTO object for transporting a single training data
Author: Paweł Bogdanowicz
"""
from datetime import datetime
from bson import ObjectId


class TrainingDataDTO:
    def __init__(self):
        self.id: ObjectId|None = None
        self.created_at: datetime | None = None
        self.branching_points: float|None = None
        self.total_length: float|None = None
        self.mean_thickness: float|None = None
        self.total_area: float|None = None
        self.is_good: bool|None = None
        self.scale: int|None = None
        self.photo_type: str|None = None


    def __json__(self):
        return {
            'id': self.id.__str__(),
            'created_at': self.created_at.__str__(),
            'branching_points': self.branching_points,
            'total_length': self.total_length,
            'mean_thickness': self.mean_thickness,
            'total_area': self.total_area,
            'is_good': self.is_good,
            'scale': self.scale,
            'photo_type': self.photo_type,
        }
