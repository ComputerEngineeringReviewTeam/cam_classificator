from json import JSONEncoder
from flask import Flask, render_template
from datetime import timedelta

from app.config.config import get_config
from app.domain.training_data.blueprints.training_data_bp import training_data_bp
from app.domain.common.authentication.blueprints.authentication_bp import authentication_bp


def _default(self, obj):
    return getattr(obj.__class__, "__json__", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


def configure_app(app):
    app.secret_key = get_config().app_secret_key
    # Register blueprints
    app.register_blueprint(training_data_bp)
    app.register_blueprint(authentication_bp)

    # Set session lifetime
    session_lifetime_days = int(get_config().session_lifetime)
    app.permanent_session_lifetime = timedelta(days=session_lifetime_days)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=session_lifetime_days)

    # Register top level routes
    @app.route('/cam')
    @app.route('/cam/')
    @app.route('/cam/index')
    @app.route('/cam/index/')
    def index():
        return render_template('base.html')

    return app


def create_app():
    app = Flask(__name__)
    return configure_app(app)


app = Flask(__name__)
