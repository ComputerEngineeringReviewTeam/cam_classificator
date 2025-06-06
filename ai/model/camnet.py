import torch
import timm
from typing import Literal

import ai.config as conf


class CamNet(torch.nn.Module):
    def __init__(self,
                 model_name,
                 mode: Literal['both', 'classifier', 'regressor'] ='both',
                 pretrained=True,
                 num_aux_inputs=1, ):
        """
        Initialize CamNet model

        Args:
            model_name (str): Name of the base model to use
            mode (str): Operating mode - 'classifier', 'regressor', or 'both'
            pretrained (bool): Whether to use pretrained weights
            num_aux_inputs (int): Number of auxiliary inputs
        """
        super(CamNet, self).__init__()

        if mode not in ['classifier', 'regressor', 'both']:
            raise ValueError("Mode must be one of: 'classifier', 'regressor', 'both'")

        self.mode = mode
        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        if mode in ['classifier', 'both']:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(conf.DROPOUT),
                torch.nn.Linear(conf.FEATURES, 1),
            )

        if mode in ['regressor', 'both']:
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
                torch.nn.ReLU(),
                torch.nn.Dropout(conf.DROPOUT),
                torch.nn.Linear(conf.FEATURES, 4),
            )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        if self.mode == 'classifier':
            return self.classifier(image_features), None
        elif self.mode == 'regressor':
            return None, self.regressor(image_features)
        else:  # mode == 'both'
            binary_output = self.classifier(image_features)
            regression_output = self.regressor(image_features)
            return binary_output, regression_output

