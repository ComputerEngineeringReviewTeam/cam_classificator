# hello.py
import random

from bson import ObjectId
from flask import Flask, render_template, request

from app.domain.training_data.queries.create_command import CreateTrainingDataCommand
from app.domain.training_data.queries.update_command import UpdateTrainingDataCommand
from app.domain.training_data.services import training_data_service
from app.form_generator.generator import *

app = Flask(__name__)

# Set a secret key for encrypting session data
app.secret_key = 'my_secret_key'


@app.route('/')
def view_form():
    form_config = load_config("form.json")
    form = create_form(form_config)


    # id = training_data_service.create(
    #     CreateTrainingDataCommand().
    #              set_total_area(random.random()).
    #              set_total_length(random.random()).
    #              set_mean_thickness(random.random()).
    #              set_branching_points(random.random())
    # )

    print(training_data_service.update(
        UpdateTrainingDataCommand().
            set_total_area(random.random()).
            set_total_length(random.random()).
            set_mean_thickness(random.random()).
            set_branching_points(random.random()).
            set_id(ObjectId('67332346a4c976581cf1fb1f'))
    ))

    print(training_data_service.delete(ObjectId('67332346a4c976581cf1fb1f')))

    print(json.dumps(training_data_service.get_all()))

    return render_template('form.html', form_html=str(form))


@app.route('/handle_post', methods=['POST'])
def handle_post():
    if request.method == 'POST':
        photo = request.files['photo']

        id = training_data_service.create(
            CreateTrainingDataCommand().
            set_total_area(random.random()).
            set_total_length(random.random()).
            set_mean_thickness(random.random()).
            set_branching_points(random.random()),
            photo
        )

        form_config = load_config("form.json")
        form_results = request.form
        typecasted_form_results = update_types(form_results, form_config)
        # just to show that type are correct
        form_content_str = ""
        for k, v in typecasted_form_results.items():
            form_content_str += f"<p>{k}={v} ({type(v).__name__})</p></br>"
        return render_template('form_result.html', form_content=form_content_str, ok=True)
    else:
        return render_template('form_result.html', ok=False)


if __name__ == '__main__':
    app.run()
