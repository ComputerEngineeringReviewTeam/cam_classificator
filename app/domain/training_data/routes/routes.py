from flask import render_template, request, redirect, jsonify, abort, url_for

from app.domain.training_data.blueprints.training_data_bp import training_data_bp
from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.domain.training_data.forms.training_data_form import TrainingDataForm


@training_data_bp.route('/all', methods=['GET'])
def view_all_data():
    all_data = training_data_service.get_all().training_data
    return render_template('view_all_data.html', data=all_data)


@training_data_bp.route('/<_id>', methods=['GET'])
def view_datapoint(_id):
    datapoint = training_data_service.get(_id)
    if datapoint is None:
        return render_template('404.html', message='Wrong ID'), 404
    else:
        return render_template('view_single_datapoint.html', datapoint=datapoint)


@training_data_bp.route('/new', methods=['GET'])
def view_new_datapoint_form():
    form = TrainingDataForm()
    return render_template("new_datapoint_form.html", form=form)


@training_data_bp.route('/new', methods=['POST'])
def publish_new_datapoint_form():
    form = TrainingDataForm()
    if form.validate_on_submit():
        data = CreateTrainingDataCommand()
        data.set_total_area(form.total_area.data)
        data.set_total_length(form.total_length.data)
        data.set_mean_thickness(form.mean_thickness.data)
        data.set_branching_points(form.branching_points.data)

        photo = form.photo.data

        training_data_service.create(data, photo)

        return render_template('upload_success.html')
    return render_template("new_datapoint_form.html", form=form)
