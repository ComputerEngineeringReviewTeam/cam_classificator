import torchmetrics

import ai.config as conf
from ai.metrics.metric_collector import CamMetricCollector
from ai.metrics.metrics import *

class MetricsList:
    class Classifier:
        std = [
            BinaryMetrics(),
            torchmetrics.classification.BinaryAccuracy(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            torchmetrics.classification.BinaryRecall(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            torchmetrics.classification.BinaryPrecision(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
        ]
    class Regressor:
        std = [
            torchmetrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            torchmetrics.regression.MeanAbsoluteError().to(device=conf.DEVICE),  # average error
            torchmetrics.regression.NormalizedRootMeanSquaredError().to(device=conf.DEVICE), # standard deviation
            torchmetrics.regression.R2Score().to(device=conf.DEVICE),    # ~accuracy
            AverageRelativeError(),
        ]

class MetricsCollections:
    std = CamMetricCollector(
        mode=conf.MODE,
        classification_metrics=MetricsList.Classifier.std,
        regression_metrics=MetricsList.Regressor.std,
    )
    only_classifier = CamMetricCollector(
        mode='classifier',
        classification_metrics=MetricsList.Classifier.std,
        regression_metrics=None
    )
    only_regressor = CamMetricCollector(
        mode='regressor',
        classification_metrics=None,
        regression_metrics=MetricsList.Regressor.std
    )


