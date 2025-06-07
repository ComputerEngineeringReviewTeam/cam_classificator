import io
import math
import random
from time import sleep

from PIL import Image

from .dto import ClassificationResult, SegmentData

SEGMENT_WIDTH = 224
SEGMENT_HEIGHT = 224


def classificate_image(image_bytes: bytes) -> ClassificationResult:
    """
    Processes the incoming image data and returns an analysis DTO.

    This is a mock service. Replace the logic inside this function with your
    actual image analysis implementation.

    Args:
        image_bytes: The raw bytes of the image file.

    Returns:
        An ClassificationResult DTO containing the analysis data.
    """
    print(f"Analyzing image of size {len(image_bytes)} bytes...")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image_width, image_height = img.size
    except Exception as e:
        # If Pillow cannot open the file, it's not a valid image format
        print(f"Error opening image: {e}")
        raise ValueError("Invalid image data provided.")

    num_cols = math.ceil(image_width / SEGMENT_WIDTH)
    num_rows = math.ceil(image_height / SEGMENT_HEIGHT)
    total_segments = num_cols * num_rows

    sleep(3)
    mock_segments = []
    for _ in range(total_segments):
        segment = SegmentData(
            branching_point=random.uniform(0.5, 5.0),
            total_length=random.uniform(100.0, 1000.0),
            mean_thickness=random.uniform(1.0, 10.0),
            total_area=random.uniform(500.0, 5000.0),
            is_good=random.choice([True, False])
        )
        mock_segments.append(segment)

    result = ClassificationResult(
        segment_width=SEGMENT_WIDTH,
        segment_height=SEGMENT_HEIGHT,
        num_cols=num_cols,
        num_rows=num_rows,
        segments=mock_segments
    )

    return result