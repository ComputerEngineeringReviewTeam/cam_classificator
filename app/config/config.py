"""
About: Config is a convenient way for accessing .env environment variables.
    When adding additional values to .env file we have to update this config too for it to see the environment variable added
Author: Paweł Bogdanowicz
"""

import os
import dotenv

__config = None


class Config:
    def __init__(self):
        if not dotenv.load_dotenv():
            raise Exception('.env file not found')

        self.mongo_url = os.getenv('MONGO_URL')
        if self.mongo_url is None:
            raise RuntimeError('Missing environment variable: MONGO_URL')

        self.photo_dict = os.getenv('PHOTO_DICT')
        if self.photo_dict is None:
            raise RuntimeError('Missing environment variable: PHOTO_DICT')

        self.app_secret_key = os.getenv('SECRET_KEY')
        if self.app_secret_key is None:
            raise RuntimeError('Missing environment variable: SECRET_KEY')


def get_config() -> Config:
    global __config
    if __config is None:
        __config = Config()
    return __config