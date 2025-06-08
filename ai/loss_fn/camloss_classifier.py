import torch


class CamLossClassifier(torch.nn.Module):
    def __init__(self):
        super(CamLossClassifier, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        binary_loss = self.bce_loss(binary_output, binary_target)
        return binary_loss
