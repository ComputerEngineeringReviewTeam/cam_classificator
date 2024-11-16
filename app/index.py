from flask import Flask, render_template, request, redirect, url_for, session
from functools import wraps

from app.config.config import get_config
from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.domain.training_data.forms.training_data_form import TrainingDataForm

app = Flask(__name__)
app.secret_key = get_config().app_secret_key


def authenticated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'authenticated' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)

    return wrapper


@app.route('/')
@app.route('/index')
def index():
    return render_template('base.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['key'] == get_config().api_key:
            session['authenticated'] = True
            return redirect(url_for('view_all_data'))
        else:
            error_msg = 'Invalid password'
        return render_template('login.html', error=error_msg)
    return render_template('login.html')


@app.route('/logout')
@authenticated
def logout():
    session.pop('auth_key', None)
    return redirect(url_for('index'))


@app.route('/training_data')
@authenticated
def view_all_data():
    all_data = training_data_service.get_all().training_data
    return render_template('view_all_data.html', data=all_data)


@app.route('/training_data/<_id>')
@authenticated
def view_datapoint(_id):
    datapoint = training_data_service.get(_id)
    if datapoint is None:
        return render_template('404.html', message='Wrong ID'), 404
    else:
        return render_template('view_single_datapoint.html', datapoint=datapoint)


@app.route('/training_data/new', methods=['GET', 'POST'])
@authenticated
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
