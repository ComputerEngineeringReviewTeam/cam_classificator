from flask import render_template, session, redirect, url_for

from data_acquisition.app.config.config import get_config
from data_acquisition.app.domain.common.authentication.blueprints.authentication_bp import authentication_bp
from data_acquisition.app.domain.common.authentication.forms.login_form import LoginForm


@authentication_bp.route('/login', methods=['GET'])
def view_login_form():
    form = LoginForm()
    return render_template('login.html', form=form)


@authentication_bp.route('/login', methods=['POST'])
def login():
    form = LoginForm()
    error = None
    if form.validate_on_submit():
        if form.key.data == get_config().api_key:
            session['authenticated'] = True
            session.permanent = True
            return redirect(url_for('index'))

        error = 'Invalid key'
    return render_template('login.html', form=form, error=error)


@authentication_bp.route('/logout', methods=['POST'])
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('index'))
