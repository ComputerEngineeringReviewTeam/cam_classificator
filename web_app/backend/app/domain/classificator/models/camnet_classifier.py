import torch
import timm


class CamNetClassifier(torch.nn.Module):
    def __init__(self,
                 model_name,
                 pretrained=True,
                 num_aux_inputs=0,
                 features=256,
                 dropout=0.2,):
        super(CamNetClassifier, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, features),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(features, 1),
        )

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        binary_output = self.classifier(image_features)
        return binary_output, torch.zeros(binary_output.shape())
