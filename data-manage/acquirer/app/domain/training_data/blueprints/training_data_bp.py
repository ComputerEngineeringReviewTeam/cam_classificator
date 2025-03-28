from flask import Blueprint

training_data_bp = Blueprint(name='training_data_bp',
                             import_name=__name__,
                             url_prefix='/cam/training_data',
                             template_folder='../templates',
                             static_folder='../static')

from data_acquisition.app.domain.training_data.routes import routes