import torch
import timm


class CamNetRegressor(torch.nn.Module):
    """
    A neural network model for regression tasks with optional auxiliary inputs.

    This class defines a regression model using the PyTorch framework. It utilizes
    the TIMM library for the initial feature extraction stage and appends a custom
    fully connected regressor to process the extracted features. The model combines
    image features and auxiliary inputs for its predictions. It is designed to
    handle regression problems where additional feature inputs may be required
    beyond raw image data.

    Attributes:
        feature_extractor (torch.nn.Module): The feature extraction model obtained
            from the TIMM library, configured based on the provided model name.
        regressor (torch.nn.Sequential): A custom regressor that processes the
            combined feature representation and auxiliary inputs to output
            predictions.
    """
    def __init__(self,
                 model_name,
                 pretrained=True,
                 num_aux_inputs=1,
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
