"""
About: Service for training data
Author: Paweł Bogdanowicz
"""

from bson import ObjectId
from werkzeug.datastructures import FileStorage

from data_acquisition.app.domain.training_data.dto.all_training_data_dto import AllTrainingDataDTO
from data_acquisition.app.domain.training_data.dto.training_data_dto import TrainingDataDTO
from data_acquisition.app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from data_acquisition.app.domain.training_data.queries.update_command import UpdateTrainingDataCommand
from data_acquisition.app.domain.training_data.repositories import training_data_repository


def get(id: ObjectId) -> TrainingDataDTO|None:
    return training_data_repository.get(id)


def get_all() -> AllTrainingDataDTO:
    return training_data_repository.get_all()


def create(command: CreateTrainingDataCommand, photo: FileStorage) -> ObjectId:
    id = training_data_repository.create(command)

    # Saving the photo
    training_data_repository.save_photo(photo, id)
    return id



def update(command: UpdateTrainingDataCommand) -> bool:
    return training_data_repository.update(command)


def delete(id: ObjectId) -> bool:
    training_data_repository.delete_photo(id)
    return training_data_repository.delete(id)
