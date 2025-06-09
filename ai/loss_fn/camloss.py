import torch

from ai.config import Modes

class CamLoss(torch.nn.Module):
    def __init__(self, mode: Modes = Modes.BOTH):
        super(CamLoss, self).__init__()

        self.mode = mode

        if mode not in Modes:
            raise ValueError("Mode must be one of: Modes.CLASSIFIER, Modes.REGRESSOR, Modes.BOTH")

        if mode in [Modes.CLASSIFIER, Modes.BOTH]:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if mode in [Modes.REGRESSOR, Modes.BOTH]:
            self.mse_loss = torch.nn.MSELoss()

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        total_loss = 0
        if self.mode in [Modes.CLASSIFIER, Modes.BOTH]:
            binary_loss = self.bce_loss(binary_output, binary_target)
            total_loss += binary_loss
        if self.mode in [Modes.REGRESSOR, Modes.BOTH]:
            regression_loss = self.mse_loss(regression_output, regression_target)
            total_loss += regression_loss

        return total_loss
