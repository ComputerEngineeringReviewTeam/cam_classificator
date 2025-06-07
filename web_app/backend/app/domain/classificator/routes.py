from flask import request, jsonify
from flask_login import login_required
from . import classificator_bp
from .service import classificate_image as service_classificate_image


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@classificator_bp.route('/', methods=['POST'])
@login_required
def classificate_image():
    """
    Receives an image file, sends it for analysis, and returns the results.
    The image should be sent as multipart/form-data with the key 'image'.
    """
    if 'image' not in request.files:
        return jsonify(message="No image was sent"), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify(message="No image was sent"), 400

    if allowed_file(file.filename):
        image_bytes = file.read()
        try:
            analysis_result_dto = service_classificate_image(
                image_bytes=image_bytes,
            )

            return jsonify(analysis_result_dto.model_dump()), 200

        except Exception as e:
            print(f"Image analysis failed for {file.filename}: {e}")
            return jsonify(message="An error occurred during image analysis."), 500
    else:
        return jsonify(message="File type not allowed."), 400