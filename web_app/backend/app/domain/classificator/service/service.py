import io
import math
import random
from time import sleep

from PIL import Image

from ..dto import ClassificationResult, SegmentData

SEGMENT_WIDTH = 224
SEGMENT_HEIGHT = 224


def classificate_image(image_bytes: bytes) -> ClassificationResult:
    """
    ---------------------------------------------------- ------------------------------------------------------
    This entry function for the classificator AI model. Change this however you see fit, just make sure, that
    it returns a proper ClassificationResult object.

    In case of an error, just throw an exception with description. It will be printed out in the console,
    and user will be informed about an unexpected error.

    All model files should be placed in this directory or new subdirectories.
    ---------------------------------------------------- ------------------------------------------------------
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image_width, image_height = img.size
    except Exception as e:
        # If Pillow cannot open the file, it's not a valid image format
        print(f"Error opening image: {e}")
        raise ValueError("Invalid image data provided.")

    num_cols = math.ceil(image_width / SEGMENT_WIDTH)
    num_rows = math.ceil(image_height / SEGMENT_HEIGHT)
    overflowed_segment_width = math.ceil((num_cols * SEGMENT_WIDTH) - image_width)
    overflowed_segment_height = math.ceil((num_rows * SEGMENT_HEIGHT) - image_height)
    total_segments = num_cols * num_rows

    branching_point_sum = 0
    num_of_good = 0

    sleep(1)
    mock_segments = []
    for _ in range(total_segments):
        segment = SegmentData(
            branching_point=int(random.uniform(1, 300)),
            is_good=random.choice([True, False])
        )
        branching_point_sum += segment.branching_point
        if segment.is_good:
            num_of_good += 1
        mock_segments.append(segment)

    result = ClassificationResult(
        branching_point_sum=branching_point_sum,
        is_good_percent=(num_of_good / total_segments) * 100,
        segment_width=SEGMENT_WIDTH,
        segment_height=SEGMENT_HEIGHT,
        overflowed_segment_width=overflowed_segment_width,
        overflowed_segment_height=overflowed_segment_height,
        num_cols=num_cols,
        num_rows=num_rows,
        segments=mock_segments
    )

    return result
