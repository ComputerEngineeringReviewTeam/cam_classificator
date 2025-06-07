from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
from flask_bcrypt import Bcrypt

db = SQLAlchemy()
login_manager = LoginManager()
cors = CORS()
bcrypt = Bcrypt()

login_manager.login_view = 'auth.login'


@login_manager.unauthorized_handler
def unauthorized():
    from flask import jsonify
    return jsonify(message="Unauthorized: Please log in."), 401

