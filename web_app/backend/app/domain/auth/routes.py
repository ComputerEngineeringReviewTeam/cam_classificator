from flask import request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from . import auth_bp
from .models import User
from ...extensions import login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify(message="Username and password required"), 400

    user = User.query.filter_by(username=username).first()

    if user and user.check_password(password):
        login_user(user)
        return jsonify(message="Login successful", user={'id': user.id, 'username': user.username, 'is_admin': user.is_admin}), 200
    return jsonify(message="Invalid credentials"), 401


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify(message="Logout successful"), 200


@auth_bp.route('/status', methods=['GET'])
def status():
    if current_user.is_authenticated:
        return jsonify(
            logged_in=True,
            user={'id': current_user.id, 'username': current_user.username, 'is_admin': current_user.is_admin}
        ), 200
    else:
        return jsonify(logged_in=False), 200