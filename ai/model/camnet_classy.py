import torch
import timm

import ai.config as conf


class CamNetClassyRegressor(torch.nn.Module):
    def __init__(self, model_name, pretrained=True, num_aux_inputs=0, num_classes=1):
        super(CamNetClassyRegressor, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf.DROPOUT),
            torch.nn.Linear(conf.FEATURES, num_classes)
            # NOTE: with NLLLoss use softmax
        )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        output = self.classifier(image_features)
        return None, output
