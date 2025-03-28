from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import FloatField, SubmitField, BooleanField, IntegerField


FILE_EXT_ALLOWED = ['jpg', 'jpeg', 'png']


class TrainingDataForm(FlaskForm):
    total_area = FloatField('Total area', default=-1.0)
    total_length = FloatField('Total length', default=-1.0)
    mean_thickness = FloatField('Mean thickness', default=-1.0)
    branching_points = FloatField('Branching points', default=-1.0)
    is_good = BooleanField('Is the tissue good', default=False)
    scale = IntegerField('Scale', default=0)
    photo = FileField('Photo',
                      validators=[FileRequired(),
                                  FileAllowed(FILE_EXT_ALLOWED, 'Images ' + str(FILE_EXT_ALLOWED) + ' only!')])
    submit = SubmitField('Send data')
