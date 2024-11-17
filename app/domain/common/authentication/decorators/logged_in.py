from functools import wraps
from flask import session, redirect, url_for


def logged_in(func):
    """
    This decorator checks if the user is authenticated in session.
    If not, it redirects the user to the login page.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'authenticated' not in session:
            return redirect(url_for('authentication_bp.view_login_form'))
        return func(*args, **kwargs)

    return wrapper
