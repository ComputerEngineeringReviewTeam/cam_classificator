import os
from json import JSONEncoder
from flask import Flask, render_template

from app.config.config import get_config


# JSONEncoder fix for making it possible to marshall objects by adding __json__ function
def _default(self, obj):
    return getattr(obj.__class__, "__json__", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default


def create_app():
    """Application factory function."""
    app_instance = Flask(__name__)

    app_instance.secret_key = os.urandom(24)

    # Register blueprints

    # Register top level routes
    # Move these to a 'main' blueprint
    @app_instance.route('/cam')
    @app_instance.route('/cam/')
    @app_instance.route('/cam/index')
    @app_instance.route('/cam/index/')
    def index():
        return render_template('base.html')

    return app_instance
