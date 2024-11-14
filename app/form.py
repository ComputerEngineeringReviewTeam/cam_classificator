from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, FileField, MultipleFileField, DecimalField
from wtforms.validators import DataRequired, NumberRange


class TrainingDataForm(FlaskForm):
    total_area = FloatField('Total area', validators=[DataRequired()])
    total_length = FloatField('Total length', validators=[DataRequired()])
    mean_thickness = FloatField('Mean thickness', validators=[DataRequired()])
    branching_points = FloatField('Branching points', validators=[DataRequired()])
    photo = FileField('Photo', validators=[DataRequired()])
    submit = SubmitField('Send data')
