import torch


class CamLossRegressor(torch.nn.Module):
    def __init__(self):
        super(CamLossRegressor, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        regression_loss = self.mse_loss(regression_output, regression_target)
        regression_loss = regression_loss.mean()
        return regression_loss
