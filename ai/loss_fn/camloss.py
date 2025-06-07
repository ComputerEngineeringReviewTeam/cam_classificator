import torch
from typing import Literal

class CamLoss(torch.nn.Module):
    def __init__(self, mode: Literal['both', 'classifier', 'regressor'] ='both'):
        super(CamLoss, self).__init__()

        if mode not in ['classifier', 'regressor', 'both']:
            raise ValueError("Mode must be one of: 'classifier', 'regressor', 'both'")

        if mode in ['both', 'classifier']:
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
        if mode in ['both', 'regressor']:
            self.mse_loss = torch.nn.MSELoss()

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        binary_loss = self.bce_loss(binary_output, binary_target)
        regression_loss = self.mse_loss(regression_output, regression_target)

        total_loss = binary_loss + regression_loss
        return total_loss
