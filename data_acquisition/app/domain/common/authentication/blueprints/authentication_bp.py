from flask import Blueprint

authentication_bp = Blueprint(name='authentication_bp',
                              import_name=__name__,
                              url_prefix='/cam/auth',
                              template_folder='../templates',
                              static_folder='../static')

from data_acquisition.app.domain.common.authentication.routes import routes
