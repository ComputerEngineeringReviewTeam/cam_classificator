from bson import ObjectId
from flask import render_template, send_from_directory

from app import get_config
from app.domain.training_data.blueprints.training_data_bp import training_data_bp
from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.domain.training_data.forms.training_data_form import TrainingDataForm

from app.domain.common.authentication.decorators.logged_in import logged_in


@training_data_bp.route('/all/', methods=['GET'])
@training_data_bp.route('/all', methods=['GET'])
@logged_in
def view_all_data():
    all_data = training_data_service.get_all().training_data
    sorted_data = sorted(all_data, key=lambda d: d.created_at, reverse=True)
    return render_template('view_all_data.html', data=sorted_data)


@training_data_bp.route('/<_id>/', methods=['GET'])
@training_data_bp.route('/<_id>', methods=['GET'])
@logged_in
def view_datapoint(_id):
    datapoint = training_data_service.get(ObjectId(_id))
    if datapoint is None:
        return render_template('404.html', message='Wrong ID'), 404
    else:
        return render_template('view_single_datapoint.html', datapoint=datapoint)


@training_data_bp.route('/new/', methods=['GET'])
@training_data_bp.route('/new', methods=['GET'])
@logged_in
def view_new_datapoint_form():
    form = TrainingDataForm()
    return render_template("new_datapoint_form.html", form=form)


@training_data_bp.route('/new/', methods=['POST'])
@training_data_bp.route('/new', methods=['POST'])
@logged_in
def publish_new_datapoint_form():
    form = TrainingDataForm()
    if form.validate_on_submit():
        data = CreateTrainingDataCommand()
        data.set_total_area(form.total_area.data)
        data.set_total_length(form.total_length.data)
        data.set_mean_thickness(form.mean_thickness.data)
        data.set_branching_points(form.branching_points.data)
        data.set_is_good(form.is_good.data)

        photo = form.photo.data
        data.set_photo_type(photo.filename.split(".")[-1])

        training_data_service.create(data, photo)

        return render_template('upload_success.html')
    return render_template("new_datapoint_form.html", form=form)


@training_data_bp.route('/photos/<_photo_filename>/', methods=['GET'])
@training_data_bp.route('/photos/<_photo_filename>', methods=['GET'])
@logged_in
def view_photo(_photo_filename):
    return send_from_directory(get_config().photo_dict, f'{_photo_filename}')

@training_data_bp.route('/<_id>/', methods=['DELETE'])
@training_data_bp.route('/<_id>', methods=['DELETE'])
@logged_in
def delete_photo(_id):
    success = training_data_service.delete(ObjectId(_id))
    if success:
        return '', 204
    else:
        return '', 404