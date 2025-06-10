import torch
import timm


class CamNetClassifier(torch.nn.Module):
    """
    A neural network model for binary classification tasks using an image feature extractor and auxiliary inputs.

    This class implements a PyTorch neural network model that uses a pre-trained image feature extractor
    from the timm library and an additional auxiliary input. The model combines features from the
    image extractor with auxiliary inputs to perform binary classification. It allows for customization
    of the feature dimensionality, dropout rate, and additional parameters for flexibility.

    Attributes:
        feature_extractor: A timm pre-trained model used to extract image features. Its output feature
            dimensionality depends on the specified model.
        classifier: A fully connected feedforward classification pipeline, consisting of layers for
            feature transformation and binary classification.
    """
    def __init__(self,
                 model_name,
                 pretrained = True,
                 num_aux_inputs = 0,
                 features = 256,
                 dropout = 0.2):
        super(CamNetClassifier, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, features),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(features, 1),
        )

    def forward(self, image):
        image_features = self.feature_extractor(image)

        binary_output = self.classifier(image_features)
        return binary_output, torch.zeros(binary_output.shape())
