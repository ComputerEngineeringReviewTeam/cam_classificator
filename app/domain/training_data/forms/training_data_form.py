from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import FloatField, SubmitField, BooleanField
from wtforms.validators import DataRequired


FILE_EXT_ALLOWED = ['jpg', 'jpeg', 'png']


class TrainingDataForm(FlaskForm):
    total_area = FloatField('Total area', validators=[DataRequired()])
    total_length = FloatField('Total length', validators=[DataRequired()])
    mean_thickness = FloatField('Mean thickness', validators=[DataRequired()])
    branching_points = FloatField('Branching points', validators=[DataRequired()])
    is_good = BooleanField('Is the tissue good ?', default=False)
    photo = FileField('Photo',
                      validators=[FileRequired(),
                                  FileAllowed(FILE_EXT_ALLOWED, 'Images ' + str(FILE_EXT_ALLOWED) + ' only!')])
    submit = SubmitField('Send data')
