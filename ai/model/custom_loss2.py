import torch
import torch.nn as nn


class CustomLoss2(nn.Module):
    def __init__(self):
        super(CustomLoss2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()          # do binarnej
        self.mse_loss = nn.MSELoss(reduction='none')    # do regresji
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        binary_loss = self.bce_loss(binary_output, binary_target)


        # mask = ~torch.isnan(regression_target)
        regression_loss = self.mse_loss(regression_output, regression_target)

        # regression_loss = regression_loss[mask].mean()  # TODO: implement regression
        regression_loss = regression_loss.mean()

            # print(f"binary_loss={binary_loss.item()}\tregression_loss={ regression_loss.item()}")

        total_loss = binary_loss + regression_loss
        return total_loss



        # binary_loss = self.ce_loss(binary_output, binary_target)
        # return binary_loss
