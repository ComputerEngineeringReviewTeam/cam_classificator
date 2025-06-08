import torch


class CamLossClassy(torch.nn.Module):
    def __init__(self):
        super(CamLossClassy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()
        # NOTE: different loss fn's not tested
        # self.loss = torch.nn.NLLLoss()
        # self.loss = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, binary_output, regression_output, binary_target, regression_target):
        regression_loss = self.loss(regression_output, regression_target)
        return regression_loss
