from app.hello import app

TRAINING_DATA_URL = '/training_data'

@app.route(TRAINING_DATA_URL, methods=['POST'])
def create_training_data():
    pass

# ...