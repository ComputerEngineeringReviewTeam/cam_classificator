import torch.nn as nn
import timm
import torchvision.transforms as transforms

from ai.nn.config import *


class CamNet(nn.Module):
    def __init__(self, model_name, pretrained=True, num_aux_inputs=1):
        super(CamNet, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + num_aux_inputs, FEATURES),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURES, 1),
        )

        # self.regressor = nn.Sequential(
        #     nn.Linear(feature_dim + num_aux_inputs, FEATURES),
        #     nn.ReLU(),
        #     nn.Dropout(DROPOUT),
        #     nn.Linear(FEATURES, 4),
        # )

        self.random_trans = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
        ])

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        scale = torch.unsqueeze(scale, 1)

        # combined_features = torch.cat((image_features, scale), dim=1)
        combined_features = image_features

        binary_output = self.classifier(combined_features)
        # regression_output = self.regressor(combined_features)

        # return binary_output, regression_output
        return binary_output, 0
