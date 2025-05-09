from app import create_app
from app.config.config import get_config

app = create_app()

if __name__ == '__main__':
    print('Starting the server')
    app.run(host=get_config().server_host,
            port=get_config().server_port,
            debug=get_config().debug)