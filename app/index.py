from flask import Flask, render_template, request, flash, redirect

from app.config.config import get_config
from app.domain.training_data.queries.create_command import  CreateTrainingDataCommand
from app.domain.training_data.queries.update_command import UpdateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.form import TrainingDataForm

app = Flask(__name__)
app.secret_key = get_config().app_secret_key


@app.route("/")
def index():
    return render_template('base.html')


@app.route('/new_data', methods=['GET', 'POST'])
def publish_new_training_data():
    form = TrainingDataForm()
    if request.method == 'POST' and form.validate_on_submit():
        photo = request.files['photo']

        data = CreateTrainingDataCommand()
        data.set_total_area(form.total_area.data)
        data.set_total_length(form.total_length.data)
        data.set_mean_thickness(form.mean_thickness.data)
        data.set_branching_points(form.branching_points.data)

        training_data_service.create(data, photo)

        return redirect('/view_all_data')

    return render_template("training_data_form.html", title='New training data', form=form)


@app.route('/view_all_data')
def view_all_data():
    all_data = training_data_service.get_all().training_data
    return render_template('view_all_data.html', data=all_data)


if __name__ == '__main__':
    app.run()