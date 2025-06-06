import torch
import torchmetrics as mtr

class BinaryMetrics:
    def __init__(self):
        self.true_positives_sum = 0.0
        self.true_negatives_sum = 0.0
        self.false_positives_sum = 0.0
        self.false_negatives_sum = 0.0

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        true_positives = ((target == 1) * (output > 0.5))
        true_negatives = ((target == 0) * (output < 0.5))
        false_positives = ((target == 0) * (output > 0.5))
        false_negatives = ((target == 1) * (output < 0.5))

        self.true_positives_sum += true_positives.sum().item()
        self.true_negatives_sum += true_negatives.sum().item()
        self.false_positives_sum += false_positives.sum().item()
        self.false_negatives_sum += false_negatives.sum().item()

    def precision(self) -> float:
        return self.true_positives_sum / (self.true_positives_sum + self.false_positives_sum)

    def recall(self) -> float:
        return self.true_positives_sum / (self.true_positives_sum + self.false_negatives_sum)

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)

    def accuracy(self) -> float:
        all_sum = self.true_positives_sum + self.true_negatives_sum + self.false_positives_sum + self.false_negatives_sum
        return (self.true_positives_sum + self.true_negatives_sum) / all_sum

    def reset(self):
        self.true_positives_sum = 0.0
        self.true_negatives_sum = 0.0
        self.false_positives_sum = 0.0
        self.false_negatives_sum = 0.0

    def print_stats(self):
        print(f"TP: {self.true_positives_sum} TN: {self.true_negatives_sum} "
              f"FP: {self.false_positives_sum} FN: {self.false_negatives_sum}")

    def print_metrics(self):
        print("Accuracy:", self.accuracy())
        print("Precision:", self.precision())
        print("Recall:", self.recall())
        print("F1:", self.f1())

    def compute(self):
        return (f"TP: {self.true_positives_sum} TN: {self.true_negatives_sum} "
                f"FP: {self.false_positives_sum} FN: {self.false_negatives_sum}")


class RelativeError:
    """BROKEN - DO NOT USE"""
    def __init__(self):
        self.target_sum = 0.0
        self.difference_sum = 0.0
        self.size = 0.0

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        self.target_sum += target.sum()
        # print("output", output.tolist(), end="\n")
        # print("target", target.tolist(), end="\n")
        # print("diff", (output - target).tolist(), end="\n")
        # print(torch.div((output - target), target).tolist())
        self.difference_sum += (output - target).sum()
        # print(target.size())
        self.size += target.size(dim=0)

    def compute(self):
        # print(self.difference_sum)
        # print(self.size)
        return self.difference_sum / self.size

    def reset(self):
        self.target_sum = 0.0
        self.difference_sum = 0.0


class AverageRelativeError:
    def __init__(self):
        self.outputs = None
        self.targets = None

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        if self.outputs is None:
            self.outputs = output
        else:
            self.outputs = torch.cat((self.outputs, output), dim=0)

        if self.targets is None:
            self.targets = target
        else:
            self.targets = torch.cat((self.targets, target), dim=0)

    def reset(self):
        self.outputs = None
        self.targets = None

    def compute(self):
        nonzero_mask = (self.targets != 0)
        zeros_count = self.targets.size(dim=0) - torch.count_nonzero(self.targets)
        relative_errors = torch.div(torch.abs(self.outputs[nonzero_mask] - self.targets[nonzero_mask]),
                                    self.targets[nonzero_mask])
        return f"avg. error: {relative_errors.mean()}, skipped {zeros_count} targets == 0"

class CamMetricCollector:
    """
    Class for easily storing and running multiple metrics on CamNet model.
    'classification_metrics' will be computed on the results of model's binary classification, and
    'regression_metrics' on the regression results.
    All metrics should either be torchmetrics.Metric or implement __call__, reset() and compute() methods.
    Operations on this class are run on each of the collected metrics.
    """
    def __init__(self,
                 classification_metrics: list[mtr.Metric],
                 regression_metrics: list[mtr.Metric]):
        self.classification_metrics = classification_metrics
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
