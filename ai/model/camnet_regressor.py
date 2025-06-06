import torch
import timm

import ai.config as conf


class CamNetRegressor(torch.nn.Module):
    def __init__(self, model_name, pretrained=True, num_aux_inputs=1):
        super(CamNetRegressor, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf.DROPOUT),
            torch.nn.Linear(conf.FEATURES, 1),
        )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        regression_output = self.regressor(image_features)
        return torch.zeros(regression_output.size()), regression_output
