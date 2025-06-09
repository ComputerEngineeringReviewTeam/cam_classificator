import os
import dotenv

class ConfigSettings:
    def __init__(self):
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if not dotenv.load_dotenv(dotenv_path=env_path):
            print(".env file has not been found. Using environment variables instead.")

        self.DEBUG = bool(os.getenv('DEBUG'))
        if self.DEBUG is None:
            raise Exception("ERROR: DEBUG environment variable not set")

        self.SERVER_HOST = os.getenv('SERVER_HOST')
        if self.SERVER_HOST is None:
            raise Exception("ERROR: SERVER_HOST environment variable not set")

        self.SERVER_PORT = int(os.getenv('SERVER_PORT'))
        if self.SERVER_PORT is None:
            raise Exception("ERROR: SERVER_PORT environment variable not set")

        self.SECRET_KEY = os.getenv('SECRET_KEY')
        if not self.SECRET_KEY:
            raise Exception("ERROR: SECRET_KEY environment variable not set")

        self.SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
        if not self.SQLALCHEMY_DATABASE_URI:
            raise Exception(f"ERROR: DATABASE_URL not set")

        self.SQLALCHEMY_TRACK_MODIFICATIONS = False

        self.ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
        if not self.ADMIN_USERNAME:
            raise Exception("ERROR: ADMIN_USERNAME environment variable not set")

        self.ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
        if not self.ADMIN_PASSWORD:
            raise Exception("ERROR: ADMIN_PASSWORD environment variable not set")


Config = ConfigSettings()