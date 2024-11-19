from app import configure_app, app, get_config

app = configure_app(app)

if __name__ == '__main__':
    print('Starting server')
    app.run(host=get_config().server_host, port=get_config().server_port, debug=get_config().debug)
