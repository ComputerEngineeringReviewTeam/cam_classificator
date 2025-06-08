import torch
import torchmetrics as mtr

from ai.utils.modes import Modes


class CamMetricCollector:
    """
    Class for easily storing and running multiple metrics on CamNet model.
    'classification_metrics' will be computed on the results of model's binary classification, and
    'regression_metrics' on the regression results.
    All metrics should either be torchmetrics.Metric or implement __call__, reset() and compute() methods.
    Operations on this class are run on each of the collected metrics.
    """
    def __init__(self,
                 mode = Modes.BOTH,
                 classification_metrics: list[mtr.Metric] | None = None,
                 regression_metrics: list[mtr.Metric] | None = None):
        if mode not in Modes:
            raise ValueError("Mode must be one of: Modes.CLASSIFIER, Modes.REGRESSOR, Modes.BOTH")

        if mode == Modes.REGRESSOR or classification_metrics is None:
            self.classification_metrics = []
        else:
            self.classification_metrics = classification_metrics
        if mode == Modes.CLASSIFIER or regression_metrics is None:
            self.regression_metrics = []
        else:
            self.regression_metrics = regression_metrics

    def call_classification_metrics(self, output: torch.Tensor, target: torch.Tensor):
        for metric in self.classification_metrics:
            metric(output, target)

    def call_regression_metrics(self, output: torch.Tensor, target: torch.Tensor):
        for metric in self.regression_metrics:
            metric(output, target)

    def __call__(self,
                 classification_output: torch.Tensor, regression_output: torch.Tensor,
                 classification_target: torch.Tensor, regression_target: torch.Tensor):
        for metric in self.classification_metrics:
            metric(classification_output, classification_target)
        for metric in self.regression_metrics:
            metric(regression_output, regression_target)

    def __iter__(self):
        for metric in self.classification_metrics + self.regression_metrics:
            yield metric

    def iter_classification_metrics(self):
        for metric in self.classification_metrics:
            yield metric

    def iter_regression_metrics(self):
        for metric in self.regression_metrics:
            yield metric

    def reset(self):
        for metric in self:
            metric.reset()

    def print_metrics(self):
        self.print_classification_metrics()
        self.print_regression_metrics()

    def print_classification_metrics(self):
        print("\tMetrics computed on classification results")
        for metric in self.classification_metrics:
            print(f"{metric.__class__.__name__}: {metric.compute()}")

    def print_regression_metrics(self):
        print("\tMetrics computed on regression results")
        for metric in self.regression_metrics:
            print(f"{metric.__class__.__name__}: {metric.compute()}")
