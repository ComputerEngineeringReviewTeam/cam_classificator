import torch
import timm
import torchvision.transforms as transforms

import ai.model.config as conf


class CamNet2(torch.nn.Module):
    def __init__(self, model_name, pretrained=True, num_aux_inputs=1):
        super(CamNet2, self).__init__()

        self.feature_extractor = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.feature_extractor.num_features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf.DROPOUT),
            torch.nn.Linear(conf.FEATURES, 1),
            torch.nn.Sigmoid()
        )

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + num_aux_inputs, conf.FEATURES),
            torch.nn.ReLU(),
            torch.nn.Dropout(conf.DROPOUT),
            torch.nn.Linear(conf.FEATURES, 1),
        )

        # self.random_trans = transforms.Compose([
        #     transforms.RandomRotation(30),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
        # ])

    def forward(self, inputs):
        image, scale = inputs
        image_features = self.feature_extractor(image)

        # scale = torch.unsqueeze(scale, 1) # temporary unused

        # combined_features = torch.cat((image_features, scale), dim=1)
        combined_features = image_features

        binary_output = self.classifier(combined_features)
        regression_output = self.regressor(combined_features)

        return binary_output, regression_output
        # return binary_output, torch.tensor([0])
        # return torch.tensor([0]), regression_output
