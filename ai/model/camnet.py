import torch
import timm

from ai.model.camnet_base import CamNetBase
from ai.utils.modes import Modes


class CamNet(CamNetBase):
    def __init__(self,
                 model_name,
                 mode: Modes = Modes.BOTH,
                 pretrained = True,
                 num_aux_inputs = 0,
                 features = 256,
                 dropout = 0.2):
        """
        Initializes the CamNet model, a neural network designed to accommodate two modes of operation:
        classification and regression. The model allows the integration of auxiliary inputs and is
        customizable in terms of architecture, dropout rate, and feature dimensions. The mode of
        operation determines the configuration of the network components, including classifiers and
        regressors, to ensure the model is tailored to specific tasks. Additional features include
        support for pretrained models and flexibility in choosing the primary feature extractor.

        Args:
            model_name (str): The name of the primary feature extractor model to use. Should be a
                compatible model supported by the timm library.
            mode (Modes): The operating mode for the model. Options are Modes.CLASSIFIER, Modes.REGRESSOR,
                or Modes.BOTH. Default is Modes.BOTH.
            pretrained (bool): Determines whether to use pretrained weights for the feature extractor.
                Default is True.
            num_aux_inputs (int): The number of additional input features to concatenate with the
                extracted features before passing them to downstream layers. Default is 0.
            features (int): The number of output features in the intermediate layer of the classifier/
                regressor. Default is 256.
            dropout (float): The dropout rate to apply in the intermediate layer of the classifier/
                regressor. Default is 0.2.
        """
        super().__init__()

        if mode not in Modes:
            raise ValueError("Mode must be one of: Modes.CLASSIFIER, Modes.REGRESSOR, Modes.BOTH")

        self.mode = mode
        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        if mode in [Modes.CLASSIFIER, Modes.BOTH]:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, features),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(features, 1),
            )

        if mode in [Modes.REGRESSOR, Modes.BOTH]:
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, features),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(features, 1),
            )

    def forward(self, image):
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
