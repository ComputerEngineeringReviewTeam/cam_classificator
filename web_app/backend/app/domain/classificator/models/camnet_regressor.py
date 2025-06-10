import torch
import timm


class CamNetRegressor(torch.nn.Module):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 num_aux_inputs=0,
                 features=256,
                 dropout=0.2):
        super(CamNetRegressor, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, features),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(features, 1),
        )

    def forward(self, image):
        image_features = self.feature_extractor(image)

        regression_output = self.regressor(image_features)
        return torch.zeros(regression_output.size()), regression_output
