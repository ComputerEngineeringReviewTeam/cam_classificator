"""
About: MongoDB connection singleton object.
Author: Paweł Bogdanowicz
"""

from data_acquisition.app.config.config import get_config
from pymongo import MongoClient

__mongo_db = None

DATABASE_NAME = 'cam-classificator'
TRAIN_DATA_COLLECTION = 'training-data'

class MongoDB:
    def __init__(self):
        self._client = MongoClient(get_config().mongo_url)
        self._db = self._client[DATABASE_NAME]

        # Defining collections
        self.training_data = self._db[TRAIN_DATA_COLLECTION]


def get_mongo() -> MongoDB:
    global __mongo_db
    if __mongo_db is None:
        __mongo_db = MongoDB()
    return __mongo_db