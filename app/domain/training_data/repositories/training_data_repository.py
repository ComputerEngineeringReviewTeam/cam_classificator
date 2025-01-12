"""
About: Repository for training data
Author: Paweł Bogdanowicz
"""
import os
from importlib.metadata import pass_none

from bson import ObjectId
from werkzeug.datastructures import FileStorage

from app.config.config import get_config
from app.domain.common.mongodb.mongo import get_mongo
from app.domain.training_data.dto.all_training_data_dto import AllTrainingDataDTO
from app.domain.training_data.dto.training_data_dto import TrainingDataDTO
from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.queries.update_command import UpdateTrainingDataCommand


def get(id: ObjectId) -> TrainingDataDTO|None:
    document = get_mongo().training_data.find_one({'_id': id})
    if document is None:
        return None

    result = TrainingDataDTO()
    result.id = document.get('_id')
    result.created_at = result.id.generation_time
    result.branching_points = document.get('branching_points')
    result.total_length = document.get('total_length')
    result.mean_thickness = document.get('mean_thickness')
    result.total_area = document.get('total_area')
    result.is_good = document.get('is_good')
    result.photo_type = document.get('photo_type')

    return result

def get_all() -> AllTrainingDataDTO:
    documents = get_mongo().training_data.find()
    results = AllTrainingDataDTO()

    for document in documents:
        result = TrainingDataDTO()
        result.id = document.get('_id')
        result.created_at = result.id.generation_time
        result.branching_points = document.get('branching_points')
        result.total_length = document.get('total_length')
        result.mean_thickness = document.get('mean_thickness')
        result.total_area = document.get('total_area')
        result.is_good = document.get('is_good')
        result.photo_type = document.get('photo_type')

        results.training_data.append(result)

    return results


def create(command: CreateTrainingDataCommand) -> ObjectId:
    result = get_mongo().training_data.insert_one(command.to_mongo_json())
    return result.inserted_id


def save_photo(photo: FileStorage, id: ObjectId):
    os.makedirs(get_config().photo_dict, exist_ok=True)

    filename = f'{id.__str__()}.{photo.filename.split(".")[-1]}'
    photo.save(os.path.join(get_config().photo_dict, filename))


def delete_photo(id: ObjectId):
    photo_data = get(id)
    if photo_data is None:
        return

    filename = f'{id}.{photo_data.photo_type}'
    try:
        os.remove(os.path.join(get_config().photo_dict, filename))
    except FileNotFoundError:
        pass
    return


def update(command: UpdateTrainingDataCommand) -> bool:
    result = get_mongo().training_data.update_one(
        {'_id': command.get_id()},
        {'$set': command.to_mongo_json()}
    )

    if result.matched_count == 0:
        return False
    return True


def delete(id: ObjectId) -> bool:
    result = get_mongo().training_data.delete_one({'_id': id})
    if result.deleted_count == 0:
        return False
    return True