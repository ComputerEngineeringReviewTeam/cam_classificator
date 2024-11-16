from flask import Flask, render_template, request, redirect, jsonify, abort, url_for
from bson import ObjectId
import json

from app.config.config import get_config
from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.domain.training_data.forms.training_data_form import TrainingDataForm

app = Flask(__name__)
app.secret_key = get_config().app_secret_key


@app.route('/')
@app.route('/index')
def index():
    return render_template('base.html')


@app.route('/training_data')
def view_all_data():
    all_data = training_data_service.get_all().training_data
    return render_template('view_all_data.html', data=all_data)


@app.route('/training_data/<_id>')
def view_datapoint(_id):
    datapoint = training_data_service.get(_id)
    if datapoint is None:
        return render_template('404.html', message='Wrong ID'), 404
    else:
        return render_template('view_single_datapoint.html', datapoint=datapoint)


@app.route('/training_data/new', methods=['GET', 'POST'])
def publish_new_training_data():
    form = TrainingDataForm()
    if request.method == 'POST' and form.validate_on_submit():
        data = CreateTrainingDataCommand()
        data.set_total_area(form.total_area.data)
        data.set_total_length(form.total_length.data)
        data.set_mean_thickness(form.mean_thickness.data)
        data.set_branching_points(form.branching_points.data)

        photo = form.photo.data

        training_data_service.create(data, photo)

        return render_template('upload_success.html')

    return render_template("training_data_form.html", form=form)


if __name__ == '__main__':
    app.run()
