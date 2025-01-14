"""
About: A DTO object for transporting training data
Author: Paweł Bogdanowicz
"""

from data_acquisition.app.domain.training_data.dto.training_data_dto import TrainingDataDTO


class AllTrainingDataDTO:
    def __init__(self):
        self.training_data: list[TrainingDataDTO] = []


    def __json__(self):
        return self.__dict__