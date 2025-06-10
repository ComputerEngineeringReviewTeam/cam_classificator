import io
import math
import random
from time import sleep
from PIL import Image
import torchvision.transforms as tf
import torch
import logging

from ..dto import ClassificationResult, SegmentData
from ..prep.split import split_image
from ..models import ModelWrapper, CamNet, CamNetRegressor, FragmentClassifier, Modes, wrap_model
from ..saved_models import SavedModelsPaths
from ..prep.normalize import normalize_minmax


def classificate_image(image_bytes: bytes) -> ClassificationResult:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        image_width, image_height = img.size
    except Exception as e:
        # If Pillow cannot open the file, it's not a valid image format
        print(f"Error opening image: {e}")
        raise ValueError("Invalid image data provided.")

    num_cols = math.ceil(image_width / 224)
    num_rows = math.ceil(image_height / 224)
    overflowed_segment_width = math.ceil((num_cols * 224) - image_width)
    overflowed_segment_height = math.ceil((num_rows * 224) - image_height)
    total_segments = num_cols * num_rows

    img_size = (224, 224)
    transforms_to_use = tf.Compose([
        tf.Resize(img_size),
        tf.ToTensor(),
    ])

    def interpret_fitness(preds):
        binary_output, regression_output = preds
        if binary_output.isnan():
            return None
        elif binary_output.item() > 0.5:
            return True
        else:
            return False

    def interpret_regression(preds):
        binary_output, regression_output = preds
        if regression_output.isnan():
            return None
        else:
            res = regression_output.item()
            res = normalize_minmax(res, 1.0, 0.0, 200.0, 0.0)
            return int(res)


    fragment_classifier = wrap_model(FragmentClassifier(),
                                     SavedModelsPaths.FragmentClassifier.MODEL99,
                                     transforms=transforms_to_use,
                                     interpret_fn=lambda x: x.argmax(dim=1) != 0)    # True if good fragment
    fitness_classifier = wrap_model(CamNet(model_name="resnet18",
                                           mode=Modes.BOTH),
                                    SavedModelsPaths.CamNet.Both.CAMNET_0845,
                                    transforms=transforms_to_use,
                                    interpret_fn=interpret_fitness)
    branching_pts_regressor = wrap_model(CamNetRegressor(model_name="resnet18"),
                                         SavedModelsPaths.CamNetRegressor.REG_G_0661,
                                         transforms=transforms_to_use,
                                         interpret_fn=interpret_regression)

    fragments = []
    fit_fragments_count = 0
    unfit_fragments_count = 0
    branching_points_sum = 0
    for i, fragment in enumerate(split_image(img, 224)):
        is_fragment_useful = fragment_classifier(fragment)
        if not is_fragment_useful:
            fragments.append(SegmentData(branching_point=None, is_good=None))
            continue

        is_fragment_fit = fitness_classifier(fragment)
        if is_fragment_fit is None:
            fragments.append(SegmentData(branching_point=None, is_good=None))
            continue
        elif not is_fragment_fit:
            fragments.append(SegmentData(branching_point=None, is_good=False))
            unfit_fragments_count += 1
            continue
        fit_fragments_count += 1

        fragment_branching_pts = branching_pts_regressor(fragment)
        fragments.append(SegmentData(branching_point=fragment_branching_pts, is_good=True))
        if fragment_branching_pts is not None:
            branching_points_sum += 1


    result = ClassificationResult(
        branching_point_sum=branching_points_sum,
        is_good_percent=(fit_fragments_count / (unfit_fragments_count + fit_fragments_count)) * 100,
        segment_width=224,
        segment_height=224,
        overflowed_segment_width=overflowed_segment_width,
        overflowed_segment_height=overflowed_segment_height,
        num_cols=num_cols,
        num_rows=num_rows,
        segments=fragments
    )

    return result
