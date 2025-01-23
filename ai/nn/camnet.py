import torch.nn as nn
import timm

from ai.nn.config import *


class CamNet(nn.Module):
    def __init__(self, model_name="resnet50", pretrained=True, num_aux_inputs=1):
        super(CamNet, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + num_aux_inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim + num_aux_inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        combined_features = torch.cat((image_features, scale), dim=1)

        binary_output = self.classifier(combined_features)
        regression_output = self.regressor(combined_features)

        return binary_output, regression_output