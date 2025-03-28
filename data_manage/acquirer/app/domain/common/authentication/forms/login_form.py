from flask_wtf import FlaskForm
from wtforms import PasswordField, SubmitField, StringField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    key = StringField('Key', validators=[DataRequired()])
    submit = SubmitField('Login')
