from flask import Flask, send_from_directory, jsonify
import os
import click

from .config import Config
from .extensions import db, migrate, login_manager, cors, bcrypt
from .domain.auth.models import User # For CLI command and shell context

REACT_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'build'))

def create_app(config_object=Config):
    app = Flask(__name__, static_folder=os.path.join(REACT_BUILD_DIR, 'static'))
    app.config.from_object(config_object)

    # # --- Your JSONEncoder Fix ---
    # def _default(self, obj):
    #     return getattr(obj.__class__, "__json__", _default.default)(obj)
    # _default.default = JSONEncoder().default
    # # Apply the monkey patch or set app.json_encoder
    # JSONEncoder.default = _default
    # Alternatively: app.json_encoder = YourCustomJSONEncoder

    # --- Initialize Extensions ---
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    cors.init_app(app, resources={r"/cam/api/*": {"origins": "*"}}, supports_credentials=True) # Example

    # --- CORS Configuration (from your run.py) ---
    # if app.config['DEBUG']: # Access DEBUG from LoadedConfig
    #     cors.init_app(app, resources={r"/cam/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
    # else:
    #     # Define your production CORS policy
    #     #cors.init_app(app, resources={r"/cam/api/*": {"origins": "*"}}, supports_credentials=True) # Example

    # --- Registering Blueprints ---
    from .domain.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/cam/api/auth')

    from .domain.admin import admin_bp
    app.register_blueprint(admin_bp, url_prefix='/cam/api/admin')


    # --- Serve React App ---
    @app.route('/cam', defaults={'path': ''})
    @app.route('/cam/', defaults={'path': ''}) # Handle trailing slash
    @app.route('/cam/<path:path>')
    def serve_react_app(path):
        if path != "" and os.path.exists(os.path.join(REACT_BUILD_DIR, path)):
            return send_from_directory(REACT_BUILD_DIR, path)
        else:
            return send_from_directory(REACT_BUILD_DIR, 'index.html')


    return app