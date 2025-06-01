from functools import wraps
from flask import jsonify
from flask_login import current_user, login_required

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            return jsonify(message="Admin access required."), 403
        return f(*args, **kwargs)
    return decorated_function