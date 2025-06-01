from app import create_app
from app.config import Config

app = create_app(Config)

if __name__ == '__main__':
    app.run(debug=Config.DEBUG,
            host=Config.SERVER_HOST,
            port=Config.SERVER_PORT)