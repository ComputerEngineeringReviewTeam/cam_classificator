from flask import Blueprint

classificator_bp = Blueprint('classificator', __name__, url_prefix='/api/classificator')

from . import routes

