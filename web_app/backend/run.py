# from app import create_app

# app = create_app()

# if __name__ == '__main__':
#     print('Starting the server')
#     app.run(host=get_config().server_host,
#             port=get_config().server_port,
#             debug=get_config().debug)


import os
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS # For development if not using proxy
from app.config.config import get_config


REACT_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'))

app = Flask(__name__,
            static_folder=os.path.join(REACT_BUILD_DIR, 'static'),
)

# This allows requests from the React dev server
if get_config().debug:
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# --- API Routes ---
@app.route('/cam/api/hello')
def hello_api():
    return jsonify(message="Hello from Flask API!")

@app.route('/cam/api/data')
def get_data():
    return jsonify(items=["item1", "item2", "item3 from Flask"])

# --- Serve React App ---
@app.route('/cam', defaults={'path': ''})
@app.route('/cam/<path:path>')
def serve_react_app(path):
    if path != "" and os.path.exists(os.path.join(REACT_BUILD_DIR, path)):
        return send_from_directory(REACT_BUILD_DIR, path)
    else:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')


if __name__ == '__main__':
    app.run(debug=get_config().debug, port=get_config().server_port)