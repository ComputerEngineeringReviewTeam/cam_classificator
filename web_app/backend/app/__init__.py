from flask import Flask, send_from_directory, jsonify
import os
import click

from .config import Config
from .extensions import db, login_manager, cors, bcrypt
from .domain.auth.models import User

from flask_migrate import upgrade as migrate_upgrade, stamp as migrate_stamp
from sqlalchemy import inspect as sqlalchemy_inspect

REACT_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'frontend', 'build'))

def create_app(config_object=Config):
    app = Flask(__name__, static_folder=os.path.join(REACT_BUILD_DIR, 'static'))
    app.config.from_object(config_object)

    # --- Initialize Extensions ---
    db.init_app(app)
    login_manager.init_app(app)
    bcrypt.init_app(app)
    cors.init_app(app, resources={r"/cam/api/*": {"origins": "*"}}, supports_credentials=True) # Example

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

    _initialize_database(app)

    return app


def _initialize_database(current_app: Flask):
    """Initialize the database. Creates all schemas and creates the admin account"""
    with current_app.app_context():
        try:
            inspector = sqlalchemy_inspect(db.engine)
            core_table_exists = inspector.has_table(User.__tablename__)

            if not core_table_exists:
                current_app.logger.info("Creating database schema from models")
                db.create_all()
                current_app.logger.info("Schema created successfully.")

        except Exception as e:
            raise Exception(f"Error during database schema setup: {e}")

        # --- Automatic Admin User Creation ---
        admin_username = Config.ADMIN_USERNAME
        admin_password = Config.ADMIN_PASSWORD

        try:
            if not User.query.filter_by(username=admin_username).first():
                current_app.logger.info(f"Attempting to create admin '{admin_username}' from environment variables...")
                admin_user = User(username=admin_username, password=admin_password, is_admin=True)
                db.session.add(admin_user)
                db.session.commit()
                current_app.logger.info(f"Admin user '{admin_username}' created successfully.")
        except Exception as e:
            db.session.rollback()
            raise Exception(f"Error creating admin user: {e}")
