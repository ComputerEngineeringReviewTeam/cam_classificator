from flask import request, jsonify
from . import admin_bp
from .decorators import admin_required
from ..auth.models import User
from ...extensions import db

@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_users():
    users = User.query.all()
    return jsonify([{'id': u.id, 'username': u.username, 'is_admin': u.is_admin} for u in users]), 200

@admin_bp.route('/users', methods=['POST'])
@admin_required
def create_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify(message="Username and password are required"), 400
    if User.query.filter_by(username=username).first():
        return jsonify(message="Username already exists"), 409

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify(message="User created successfully", user={'id': new_user.id, 'username': new_user.username, 'is_admin': new_user.is_admin}), 201

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    user_to_delete = User.query.get_or_404(user_id)

    if user_to_delete.is_admin:
        return jsonify(message="You cannot delete admin account."), 403

    db.session.delete(user_to_delete)
    db.session.commit()
    return jsonify(message="User deleted successfully"), 200