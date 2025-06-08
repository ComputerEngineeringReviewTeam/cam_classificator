import torch
import timm

from ai.config import Modes, FEATURES, DROPOUT


class CamNet(torch.nn.Module):
    def __init__(self,
                 model_name,
                 mode: Modes = Modes.BOTH,
                 pretrained=True,
                 num_aux_inputs=1):
        """
        Initialize CamNet model

        Args:
            model_name (str): Name of the base model to use
            mode (str): Operating mode - 'classifier', 'regressor', or 'both'
            pretrained (bool): Whether to use pretrained weights
            num_aux_inputs (int): Number of auxiliary inputs
        """
        super(CamNet, self).__init__()

        if mode not in Modes:
            raise ValueError("Mode must be one of: Modes.CLASSIFIER, Modes. REGRESSOR, Modes.BOTH")

        self.mode = mode
        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        if mode in [Modes.CLASSIFIER, Modes.BOTH]:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(DROPOUT),
                torch.nn.Linear(FEATURES, 1),
            )

        if mode in [Modes.REGRESSOR, Modes.BOTH]:
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(DROPOUT),
                torch.nn.Linear(FEATURES, 1),
            )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        if self.mode == Modes.CLASSIFIER:
            return self.classifier(image_features), None
        elif self.mode == Modes.REGRESSOR:
            return None, self.regressor(image_features)
        else:
            # mode == Modes.BOTH
            binary_output = self.classifier(image_features)
            regression_output = self.regressor(image_features)
            return binary_output, regression_output
