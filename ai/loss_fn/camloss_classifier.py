import torch

from ai.loss_fn.camloss_base import CamLossBase


class CamLossClassifier(CamLossBase):
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        binary_loss = self.bce_loss(binary_output, binary_target)
        return binary_loss
