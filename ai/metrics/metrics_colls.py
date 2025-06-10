import torchmetrics

import ai.config as conf
from ai.metrics.metric_collector import CamMetricCollector
from ai.metrics.metrics import *

class MetricsList:
    class Classifier:
        std = [
            BinaryMetrics(),
        ]
        full = [
            BinaryMetrics(),
            torchmetrics.classification.BinaryAccuracy(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            torchmetrics.classification.BinaryRecall(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
            torchmetrics.classification.BinaryPrecision(threshold=conf.BINARY_ACCURACY_THRESHOLD).to(device=conf.DEVICE),
        ]
    class Regressor:
        std = [
            torchmetrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            AverageRelativeError(),
            AverageError()
        ]
        full = [
            torchmetrics.regression.MeanSquaredError().to(device=conf.DEVICE),
            AverageRelativeError(),
            AverageError()
        ]

class MetricsCollections:
    std = CamMetricCollector(
        classification_metrics=MetricsList.Classifier.std,
        regression_metrics=MetricsList.Regressor.std,
    )
    only_classifier = CamMetricCollector(
        classification_metrics=MetricsList.Classifier.std,
        regression_metrics=[]
    )
    only_regressor = CamMetricCollector(
        classification_metrics=[],
        regression_metrics=MetricsList.Regressor.std
    )
