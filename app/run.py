from app import configure_app, app

app = configure_app(app)

if __name__ == '__main__':
    app.run()
