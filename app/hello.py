# app.py

from flask import Flask, render_template, request, session, redirect, url_for, flash
from app.form_generator.generator import *

app = Flask(__name__)

# Set a secret key for encrypting session data
app.secret_key = 'my_secret_key'

# Simple dummy users
users = {'biotech': 'password', 'admin': 'admin1'}


@app.route('/')
def view_form():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    form_config = load_config("form.json")
    form = create_form(form_config)

    return render_template('form.html', form_html=str(form))


@app.route('/handle_post', methods=['POST'])
def handle_post():
    if request.method == 'POST':
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


@app.route("/login", methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('view_form'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and users[username] == password:
            session['user_id'] = username
            return redirect(url_for('view_form'))
        else:
            flash("Wrong username / password")

    return render_template('login.html')


@app.route("/logout")
def logout():
    if 'user_id' in session:
        session.pop('user_id', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(port=3000)
