import torch
from torchmetrics import Metric
from typing import Tuple, Optional, Dict, Any


class BinaryMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("true_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("true_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.true_positives += ((target == 1) * (preds > 0.5)).sum()
        self.true_negatives += ((target == 0) * (preds < 0.5)).sum()
        self.false_positives += ((target == 0) * (preds > 0.5)).sum()
        self.false_negatives += ((target == 1) * (preds < 0.5)).sum()

    def compute(self) -> Dict[str, torch.Tensor]:
        # precision = self.true_positives / (self.true_positives + self.false_positives)
        # recall = self.true_positives / (self.true_positives + self.false_negatives)
        # f1 = 2 * precision * recall / (precision + recall)
        # accuracy = (self.true_positives + self.true_negatives) / (
        #         self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        # )

        return {
            "accuracy": (self.true_positives + self.true_negatives) / (
                self.true_positives + self.true_negatives + self.false_positives + self.false_negatives),
            "precision": self.true_positives / (self.true_positives + self.false_positives),
            "recall": self.true_positives / (self.true_positives + self.false_negatives),
            # "f1": 2 * precision * recall / (precision + recall),
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives
        }

    def print_stats(self):
        stats = self.compute()
        print(f"TP: {stats['true_positives']:.1f} TN: {stats['true_negatives']:.1f} "
              f"FP: {stats['false_positives']:.1f} FN: {stats['false_negatives']:.1f}")

    def print_metrics(self):
        stats = self.compute()
        print(f"Accuracy: {stats['accuracy']:.4f}")
        print(f"Precision: {stats['precision']:.4f}")
        print(f"Recall: {stats['recall']:.4f}")
        print(f"F1: {stats['f1']:.4f}")


class AverageRelativeError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("outputs", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.outputs.append(preds)
        self.targets.append(target)

    def compute(self) -> Tuple[torch.Tensor, int]:
        outputs = torch.cat(self.outputs, dim=0)
        targets = torch.cat(self.targets, dim=0)

        nonzero_mask = (targets != 0)
        zeros_count = targets.size(dim=0) - torch.count_nonzero(targets)

        relative_errors = torch.div(
            torch.abs(outputs[nonzero_mask] - targets[nonzero_mask]),
            targets[nonzero_mask]
        )

        return relative_errors.mean(), zeros_count

    def __str__(self):
        avg_error, zeros_count = self.compute()
        return f"avg. error: {avg_error}, skipped {zeros_count} targets == 0"


class RelativeError(Metric):
    """BROKEN - DO NOT USE"""
    def __init__(self):
        super().__init__()
        self.add_state("target_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("difference_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("size", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.target_sum += target.sum()
        self.difference_sum += (preds - target).sum()
        self.size += target.size(dim=0)

    def compute(self) -> torch.Tensor:
        return self.difference_sum / self.size