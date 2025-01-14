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
        # uncomment when running not by docker compose
        if not dotenv.load_dotenv():
            raise Exception('.env file not found')

        debug = os.getenv('DEBUG')
        if debug is None:
            raise RuntimeError('Missing environment variable: DEBUG')

        if str.lower(debug) == 'true':
            self.debug = True
        elif str.lower(debug) == 'false':
            self.debug = False
        else:
            raise RuntimeError(f'Invalid environment variable value: DEBUG={debug}')

        self.server_host = os.getenv('SERVER_HOST')
        if self.server_host is None:
            raise RuntimeError('Missing environment variable: SERVER_HOST')

        self.server_port = os.getenv('SERVER_PORT')
        if self.server_port is None:
            raise RuntimeError('Missing environment variable: SERVER_PORT')

        self.mongo_url = os.getenv('MONGO_URL')
        if self.mongo_url is None:
            raise RuntimeError('Missing environment variable: MONGO_URL')

        photo_dict = os.getenv('PHOTO_DICT')
        if photo_dict is None:
            raise RuntimeError('Missing environment variable: PHOTO_DICT')
        self.photo_dict = os.path.abspath(photo_dict)

        self.app_secret_key = os.getenv('SECRET_KEY')
        if self.app_secret_key is None:
            raise RuntimeError('Missing environment variable: SECRET_KEY')

        self.api_key = os.getenv('API_KEY')
        if self.api_key is None:
            raise RuntimeError('Missing environment variable: API_KEY')

def get_config() -> Config:
    global __config
    if __config is None:
        __config = Config()
    return __config