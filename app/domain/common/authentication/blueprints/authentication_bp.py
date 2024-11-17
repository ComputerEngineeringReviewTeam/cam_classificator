from flask import Blueprint

authentication_bp = Blueprint(name='authentication_bp',
                              import_name=__name__,
                              url_prefix='/auth',
                              template_folder='../templates',
                              static_folder='../static')

from app.domain.common.authentication.routes import routes
